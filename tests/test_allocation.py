"""Thruster layout, allocation matrix, limits and lag."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from rov_rl.pid_baseline import allocate
from rov_rl.vehicle import (
    NUM_THRUSTERS,
    THRUSTER_AXIS,
    THRUSTER_POS_M,
    VEHICLE,
    allocation_matrix,
    load_model,
    thruster_site_names,
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return load_model()


def test_sites_match_published_geometry(model):
    for j, name in enumerate(thruster_site_names()):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        np.testing.assert_allclose(model.site_pos[sid], THRUSTER_POS_M[j], atol=1e-4)
    tam = allocation_matrix(model)
    np.testing.assert_allclose(tam[:3].T, THRUSTER_AXIS, atol=1e-4)


def test_allocation_full_rank_and_well_conditioned(model):
    tam = allocation_matrix(model)
    assert tam.shape == (6, NUM_THRUSTERS)
    assert np.linalg.matrix_rank(tam) == 6
    s = np.linalg.svd(tam, compute_uv=False)
    assert s.min() > 0.2  # smallest singular value [N per N or N m per N]


def test_pinv_reproduces_wrench(model):
    tam = allocation_matrix(model)
    pinv = np.linalg.pinv(tam)
    wrench = np.array([10.0, -5.0, 8.0, 0.5, -0.3, 1.0])
    np.testing.assert_allclose(tam @ (pinv @ wrench), wrench, atol=1e-9)


@pytest.mark.parametrize("axis,published_n", [(0, 85.0), (1, 85.0), (2, 120.0)])
def test_max_axis_force_matches_published(model, axis, published_n):
    """Max pure force along a body axis within thruster limits vs [vB22] Table 4."""
    pinv = np.linalg.pinv(allocation_matrix(model))
    unit = np.zeros(6)
    unit[axis] = 1.0
    per_newton = pinv @ unit
    max_force = VEHICLE.thrust_max_n / np.max(np.abs(per_newton))
    assert max_force == pytest.approx(published_n, rel=0.02)


def test_allocate_saturation_preserves_direction(model):
    pinv = np.linalg.pinv(allocation_matrix(model))
    wrench = np.array([500.0, 0.0, 200.0, 0.0, 0.0, 10.0])
    thrust, saturated = allocate(pinv, wrench, VEHICLE.thrust_max_n)
    assert saturated
    assert np.max(np.abs(thrust)) == pytest.approx(VEHICLE.thrust_max_n)
    unsat = pinv @ wrench
    np.testing.assert_allclose(thrust / np.linalg.norm(thrust), unsat / np.linalg.norm(unsat), atol=1e-12)
    small = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    thrust, saturated = allocate(pinv, small, VEHICLE.thrust_max_n)
    assert not saturated
    np.testing.assert_allclose(thrust, pinv @ small)


def test_actuator_limits_and_lag_parameters(model):
    assert model.nu == NUM_THRUSTERS
    np.testing.assert_allclose(model.actuator_ctrlrange, [[-VEHICLE.thrust_max_n, VEHICLE.thrust_max_n]] * 8)
    assert np.all(model.actuator_ctrllimited == 1)
    assert np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_FILTER)
    np.testing.assert_allclose(model.actuator_dynprm[:, 0], VEHICLE.thruster_tau_s)
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_SITE)


def test_thruster_first_order_lag_and_clamp(model):
    d = mujoco.MjData(model)
    d.ctrl[:] = 100.0  # beyond the limit: must clamp to thrust_max
    tau = VEHICLE.thruster_tau_s
    steps = round(tau / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, d)
    t = steps * model.opt.timestep
    # Euler-discretized first-order step response at t ~ tau.
    expected = VEHICLE.thrust_max_n * (1.0 - (1.0 - model.opt.timestep / tau) ** steps)
    np.testing.assert_allclose(d.act, expected, rtol=1e-6)
    assert d.act[0] == pytest.approx(VEHICLE.thrust_max_n * (1.0 - math.exp(-t / tau)), rel=0.05)
    for _ in range(round(10 * tau / model.opt.timestep)):
        mujoco.mj_step(model, d)
    np.testing.assert_allclose(d.actuator_force, VEHICLE.thrust_max_n, rtol=1e-3)
