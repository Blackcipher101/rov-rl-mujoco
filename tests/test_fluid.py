"""Fluid-model, current and buoyancy sanity tests against analytic expectations."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from rov_rl import core
from rov_rl.config import RandomizationConfig, SimConfig
from rov_rl.vehicle import GRAVITY_MPS2, VEHICLE, load_model

TARGET_QUAD = np.array(VEHICLE.quadratic_damping[:3])
TARGET_LIN_SURGE = VEHICLE.linear_damping[0]


@pytest.fixture()
def md() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = load_model()
    return m, mujoco.MjData(m)


def fluid_force(m: mujoco.MjModel, d: mujoco.MjData, qvel: np.ndarray, quat: np.ndarray | None = None) -> np.ndarray:
    mujoco.mj_resetData(m, d)
    if quat is not None:
        d.qpos[3:7] = quat
    d.qvel[:] = qvel
    mujoco.mj_forward(m, d)
    return d.qfrc_fluid.copy()


def test_geoms_use_inertia_box_model(md):
    """MJX (JAX) ignores fluidshape; any ellipsoid geom would make CPU and MJX diverge."""
    m, _ = md
    assert np.all(m.geom_fluid == 0.0)
    assert m.opt.density == pytest.approx(1000.0)


def test_no_fluid_force_at_rest(md):
    """MuJoCo's fluid model contains no buoyancy: zero force for zero relative velocity."""
    m, d = md
    np.testing.assert_allclose(fluid_force(m, d, np.zeros(6)), 0.0, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_translational_damping_matches_identified(md, axis):
    """Fit F(v) = -(c_lin v + c_quad v|v|) at two speeds; compare with [vB22] Table A1."""
    m, d = md
    speeds = np.array([0.2, 0.6])
    forces = []
    for v in speeds:
        qvel = np.zeros(6)
        qvel[axis] = v
        forces.append(-fluid_force(m, d, qvel)[axis])
    a = np.stack([speeds, speeds**2], axis=1)
    c_lin, c_quad = np.linalg.solve(a, np.array(forces))
    assert c_quad == pytest.approx(TARGET_QUAD[axis], rel=1e-3)
    assert c_lin == pytest.approx(TARGET_LIN_SURGE, rel=1e-3)


def _inertia_box_expected(m: mujoco.MjModel, quat: np.ndarray, linvel: np.ndarray, angvel_body: np.ndarray,
                          wind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent re-implementation of the inertia-box equations (MuJoCo docs, MJX passive.py)."""
    rot = core.quat_to_mat(np, quat)
    force = np.zeros(3)
    torque = np.zeros(3)
    for b in range(1, m.nbody):
        mass, inertia = m.body_mass[b], m.body_inertia[b]
        box = np.sqrt(6.0 * np.array([inertia[1] + inertia[2] - inertia[0],
                                      inertia[0] + inertia[2] - inertia[1],
                                      inertia[0] + inertia[1] - inertia[2]]) / mass)
        v = rot.T @ (linvel - wind)  # all bodies share the CoM, so no lever-arm term
        w = angvel_body
        diam = box.mean()
        f = -3.0 * math.pi * diam * m.opt.viscosity * v
        f -= 0.5 * m.opt.density * np.array([box[1] * box[2], box[0] * box[2], box[0] * box[1]]) * np.abs(v) * v
        t = -math.pi * diam**3 * m.opt.viscosity * w
        t -= m.opt.density / 64.0 * np.array([box[0] * (box[1] ** 4 + box[2] ** 4),
                                              box[1] * (box[0] ** 4 + box[2] ** 4),
                                              box[2] * (box[0] ** 4 + box[1] ** 4)]) * np.abs(w) * w
        force += rot @ f
        torque += t
    return force, torque


def test_fluid_force_matches_inertia_box_equations(md):
    m, d = md
    rng = np.random.default_rng(0)
    for _ in range(5):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        linvel, angvel, wind = rng.normal(size=3) * 0.5, rng.normal(size=3), rng.normal(size=3) * 0.3
        m.opt.wind[:] = wind
        got = fluid_force(m, d, np.concatenate([linvel, angvel]), quat)
        f_exp, t_exp = _inertia_box_expected(m, quat, linvel, angvel, wind)
        np.testing.assert_allclose(got[:3], f_exp, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(got[3:], t_exp, rtol=1e-9, atol=1e-9)


def test_wind_is_medium_velocity(md):
    """option.wind = water velocity: still body in current w == body moving at -w in still water."""
    m, d = md
    w = np.array([0.3, -0.1, 0.05])
    m.opt.wind[:] = w
    f_current = fluid_force(m, d, np.zeros(6))
    m.opt.wind[:] = 0.0
    f_moving = fluid_force(m, d, np.concatenate([-w, np.zeros(3)]))
    np.testing.assert_allclose(f_current, f_moving, atol=1e-12)
    assert f_current[0] > 0.0  # current along +x pushes the vehicle along +x


def _neutral_params(**overrides: float) -> core.EpisodeParams:
    cfg = RandomizationConfig(p_station=1.0, current_const_mps=(0.0, 0.0), current_amp_mps=(0.0, 0.0),
                              station_start_pos_m=0.0, station_start_yaw_rad=0.0, start_tilt_rad=0.0,
                              buoyancy_ratio=(1.0, 1.0))
    p = core.sample_episode(np, np.full(core.NUM_UNIFORMS, 0.5), cfg, SimConfig())
    return p._replace(**overrides)


def simulate(m: mujoco.MjModel, d: mujoco.MjData, p: core.EpisodeParams, duration_s: float,
             extra_body_force_n: np.ndarray | None = None, quat: np.ndarray | None = None) -> None:
    mujoco.mj_resetData(m, d)
    if quat is not None:
        d.qpos[3:7] = quat
    rov = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rov")
    for _ in range(round(duration_s / m.opt.timestep)):
        wrench = core.buoyancy_wrench(np, d.qpos[3:7], p)
        if extra_body_force_n is not None:
            wrench[:3] += core.quat_to_mat(np, d.qpos[3:7]) @ extra_body_force_n
        d.xfrc_applied[rov] = wrench
        mujoco.mj_step(m, d)


def test_terminal_velocity_matches_analytic(md):
    """Constant 40 N surge force: steady speed solves c_quad v^2 + c_lin v = F."""
    m, d = md
    force_n = 40.0
    simulate(m, d, _neutral_params(), 15.0, extra_body_force_n=np.array([force_n, 0.0, 0.0]))
    c_q, c_l = TARGET_QUAD[0], TARGET_LIN_SURGE
    v_expected = (-c_l + math.sqrt(c_l**2 + 4.0 * c_q * force_n)) / (2.0 * c_q)
    assert d.qvel[0] == pytest.approx(v_expected, rel=5e-3)
    assert abs(d.qvel[1]) < 1e-6 and abs(d.qvel[2]) < 1e-6


def test_neutral_buoyancy_holds_depth(md):
    m, d = md
    simulate(m, d, _neutral_params(), 10.0)
    assert np.linalg.norm(d.qpos[:3]) < 1e-6
    assert np.linalg.norm(d.qvel) < 1e-6


def test_positive_buoyancy_terminal_rise_speed(md):
    """B/W = 1.01: rise speed solves Z|w| w^2 + c_lin w = (B - W)."""
    m, d = md
    ratio = 1.01
    simulate(m, d, _neutral_params(buoyancy_ratio=np.float64(ratio)), 20.0)
    net = (ratio - 1.0) * VEHICLE.mass_kg * GRAVITY_MPS2
    c_q, c_l = TARGET_QUAD[2], TARGET_LIN_SURGE
    w_expected = (-c_l + math.sqrt(c_l**2 + 4.0 * c_q * net)) / (2.0 * c_q)
    assert d.qvel[2] == pytest.approx(w_expected, rel=1e-2)


def test_righting_moment_restores_level(md):
    """CoB above CoG: a rolled vehicle returns towards level; initial torque = B z_b sin(roll)."""
    m, d = md
    p = _neutral_params()
    roll = math.radians(20.0)
    quat = core.yaw_quat(np, roll, 0.0, 0.0)
    wrench = core.buoyancy_wrench(np, quat, p)
    expected = core.buoyancy_n(p) * p.cob_offset_m[2] * math.sin(roll)
    assert wrench[3] == pytest.approx(-expected, rel=1e-9)  # restoring: opposes +roll
    simulate(m, d, p, 6.0, quat=quat)
    cos_tilt = core.quat_to_mat(np, d.qpos[3:7])[2, 2]
    assert cos_tilt > math.cos(roll / 2.0)
