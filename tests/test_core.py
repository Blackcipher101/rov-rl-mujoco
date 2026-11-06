"""Reference generation, randomization sampling, observation and actuation mapping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rov_rl import core
from rov_rl.config import DISTURBANCE_LEVELS_MPS, TRAIN_RANDOMIZATION, SimConfig, eval_randomization
from rov_rl.vehicle import VEHICLE

SIM = SimConfig()


def sample(cfg, seed: int) -> core.EpisodeParams:
    return core.sample_episode(np, np.random.default_rng(seed).random(core.NUM_UNIFORMS), cfg, SIM)


def test_station_reference_is_constant_origin():
    p = sample(eval_randomization("station", 2), 0)
    for t in (0.0, 5.0, 39.0):
        ref = core.reference(np, t, p)
        np.testing.assert_allclose(ref.pos_m, 0.0)
        np.testing.assert_allclose(ref.vel_mps, 0.0)
        assert ref.yaw_rad == 0.0 and ref.yaw_rate_radps == 0.0


def test_track_reference_kinematics():
    p = sample(eval_randomization("track", 0), 1)
    t_end = float(core.path_end_time_s(np, p))
    assert 0.0 < t_end < SIM.episode_duration_s
    np.testing.assert_allclose(core.reference(np, 0.0, p).pos_m, p.waypoints_m[0])
    np.testing.assert_allclose(core.reference(np, t_end + 1.0, p).pos_m, p.waypoints_m[-1], atol=1e-9)
    np.testing.assert_allclose(core.reference(np, t_end + 1.0, p).vel_mps, 0.0)
    dt = 1e-4
    for t in np.linspace(0.3, t_end - 0.3, 7):
        r0, r1 = core.reference(np, t, p), core.reference(np, t + dt, p)
        assert np.linalg.norm(r0.vel_mps) == pytest.approx(float(p.ref_speed_mps), rel=1e-9) or \
            np.linalg.norm(r0.vel_mps) == 0.0  # exactly at a corner
        np.testing.assert_allclose((r1.pos_m - r0.pos_m) / dt, r0.vel_mps, atol=1e-6)
        dyaw = core.wrap_angle(np, r1.yaw_rad - r0.yaw_rad) / dt
        assert dyaw == pytest.approx(float(r0.yaw_rate_radps), abs=1e-6)
        assert abs(r0.yaw_rad) <= math.pi + 1e-12


def test_eval_levels_set_exact_current_magnitude():
    for level, (const, amp) in enumerate(DISTURBANCE_LEVELS_MPS):
        p = sample(eval_randomization("station", level), level)
        assert np.linalg.norm(p.current_const_mps) == pytest.approx(const, abs=1e-12)
        np.testing.assert_allclose(p.current_amp_mps, amp * np.array([1.0, 1.0, 0.3]))
        assert p.mass_scale == 1.0 and p.drag_scale == 1.0
        assert not np.any(p.thruster_failed)


def test_sampling_respects_ranges_and_is_deterministic():
    cfg = TRAIN_RANDOMIZATION
    for seed in range(50):
        p = sample(cfg, seed)
        assert cfg.mass_scale[0] <= p.mass_scale <= cfg.mass_scale[1]
        assert cfg.buoyancy_ratio[0] <= p.buoyancy_ratio <= cfg.buoyancy_ratio[1]
        assert cfg.drag_scale[0] <= p.drag_scale <= cfg.drag_scale[1]
        assert cfg.thruster_tau_s[0] <= p.thruster_tau_s <= cfg.thruster_tau_s[1]
        assert np.all((p.thruster_gain >= cfg.thruster_gain[0]) & (p.thruster_gain <= cfg.thruster_gain[1]))
        assert np.linalg.norm(p.current_const_mps) <= cfg.current_const_mps[1] + 1e-12
        assert p.thruster_failed.sum() in (0.0, 1.0)
        assert np.linalg.norm(p.start_quat) == pytest.approx(1.0)
    a, b = sample(cfg, 3), sample(cfg, 3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_thruster_failure_scenario_and_ctrl_mapping():
    p = sample(eval_randomization("station", 2, "thruster_failure"), 5)
    assert p.thruster_failed.sum() == 1.0
    assert p.failure_time_s == pytest.approx(0.25 * SIM.episode_duration_s)
    action = np.full(8, 0.5)
    before = core.thruster_ctrl(np, action, p.failure_time_s - 0.01, p)
    after = core.thruster_ctrl(np, action, p.failure_time_s, p)
    np.testing.assert_allclose(before, 0.5 * VEHICLE.thrust_max_n)
    np.testing.assert_allclose(after, 0.5 * VEHICLE.thrust_max_n * (1.0 - p.thruster_failed))
    np.testing.assert_allclose(core.thruster_ctrl(np, np.full(8, 3.0), 0.0, p), VEHICLE.thrust_max_n)


def test_observation_shape_frame_and_sanitization():
    qpos = np.concatenate([[1.0, 0.0, 0.0], core.yaw_quat(np, 0.0, 0.0, math.pi / 2)])
    qvel = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.1])
    ref = core.Reference(np.zeros(3), np.zeros(3), np.float64(math.pi / 2), np.float64(0.0))
    obs = core.observation(np, qpos, qvel, ref, np.zeros(3), np.zeros(8), SIM)
    assert obs.shape == (core.OBS_SIZE,)
    # Target 1 m away along world -x; vehicle faces world +y -> target is to its left (body +y).
    np.testing.assert_allclose(obs[:3], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(obs[3:6], [0.2, 0.0, 0.0], atol=1e-12)  # world +y velocity = body forward
    np.testing.assert_allclose(obs[12:15], [0.0, 0.0, -1.0], atol=1e-12)
    bad = core.observation(np, qpos * np.nan, qvel, ref, np.zeros(3), np.zeros(8), SIM)
    assert np.all(np.isfinite(bad))


def test_integral_is_leaky_and_clamped():
    i = np.zeros(3)
    for _ in range(10_000):
        i = core.update_integral(np, i, np.array([10.0, -10.0, 0.0]), SIM)
    np.testing.assert_allclose(i, [SIM.integral_limit_ms, -SIM.integral_limit_ms, 0.0])
    i = core.update_integral(np, np.array([1.0, 0.0, 0.0]), np.zeros(3), SIM)
    assert i[0] == pytest.approx(1.0 - SIM.ctrl_dt_s / SIM.integral_tau_s)


def test_current_velocity_profile():
    p = sample(eval_randomization("station", 3), 4)
    ts = np.linspace(0.0, 40.0, 401)
    c = np.stack([core.current_velocity(np, t, p) for t in ts])
    np.testing.assert_allclose(c.mean(axis=0), p.current_const_mps, atol=0.06)
    assert np.max(np.abs(c - p.current_const_mps)) <= np.max(p.current_amp_mps) + 1e-12
