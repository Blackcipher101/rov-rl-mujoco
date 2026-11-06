"""Nav CPU environment: reset, observation, noise, collision, success and timeout."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rov_rl import core, nav_core
from rov_rl.config import NAV_TRAIN_SCENE, NAV_TRAIN_VEHICLE, NavConfig, SimConfig, SonarConfig, nav_eval_vehicle
from rov_rl.nav_cpu_env import NavCpuEnv
from rov_rl.vehicle import load_model

CFG = NavConfig()
NOISELESS = dataclasses.replace(CFG, sonar=dataclasses.replace(SonarConfig(), range_noise_std_m=0.0,
                                                               range_noise_frac=0.0, dropout_prob=0.0,
                                                               dvl_noise_std_mps=0.0))


def env(cfg: NavConfig = CFG, vehicle=NAV_TRAIN_VEHICLE) -> NavCpuEnv:
    return NavCpuEnv(vehicle, NAV_TRAIN_SCENE, cfg)


def straight_line_episode(e: NavCpuEnv, goal_x_m: float, pillar_x_m: float | None) -> nav_core.NavEpisode:
    ep = e.sample_params(np.random.default_rng(0))
    s = ep.scene
    active = np.zeros_like(s.cap_active)
    cap_a, cap_b = s.cap_a_m.copy(), s.cap_b_m.copy()
    if pillar_x_m is not None:
        active[0] = 1.0
        cap_a[0] = [pillar_x_m, 0.0, -5.0]
        cap_b[0] = [pillar_x_m, 0.0, 30.0]
    scene = s._replace(goal_m=np.array([goal_x_m, 0.0, 0.0]), seabed_z_m=np.float64(-4.0), cap_a_m=cap_a,
                       cap_b_m=cap_b, cap_radius_m=np.full_like(s.cap_radius_m, 0.4), cap_active=active,
                       box_active=np.zeros_like(s.box_active))
    return nav_core.NavEpisode(vehicle=ep.vehicle._replace(start_quat=core.yaw_quat(np, 0.0, 0.0, 0.0)), scene=scene)


def test_reset_deterministic_and_obs_valid():
    e = env()
    a, b = e.reset(seed=11), e.reset(seed=11)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (nav_core.nav_obs_size(CFG.sonar),) == (e.observation_size,)
    assert np.all(np.isfinite(a))
    obs, reward, done, info = e.step(np.zeros(4))
    assert np.all(np.isfinite(obs)) and np.isfinite(reward) and not done
    assert info.t_s == pytest.approx(CFG.sim.ctrl_dt_s)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        env().step(np.zeros(4))


def test_physics_model_is_the_unchanged_vehicle():
    assert env().model.ngeom == load_model().ngeom


def test_noise_seed_controls_noise_and_noiseless_matches_truth():
    e = env()
    p = e.sample_params(np.random.default_rng(4))
    a = e.reset(params=p, noise_seed=1)
    b = e.reset(params=p, noise_seed=1)
    c = e.reset(params=p, noise_seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    clean = env(NOISELESS)
    clean.reset(params=p, noise_seed=1)
    fan, vertical = nav_core.true_ranges(np, clean.data.qpos, p.scene, NOISELESS.sonar)
    np.testing.assert_array_equal(clean.measurement.sonar_range_m, fan)
    assert float(clean.measurement.altimeter_m) == float(vertical[0])


def test_driving_into_a_pillar_ends_in_collision():
    e = env(vehicle=nav_eval_vehicle(0))
    obs = e.reset(params=straight_line_episode(e, goal_x_m=20.0, pillar_x_m=5.0), noise_seed=0)
    done, steps = False, 0
    while not done:
        obs, reward, done, info = e.step(np.array([1.0, 0.0, 0.0, 0.0]))
        steps += 1
    assert info.collision and not info.success and not info.timeout
    assert reward < -CFG.reward.collision_penalty + 1.0
    # Hit about 5 - 0.4 - 0.42 m ahead; the bounding-sphere surface reached the pillar.
    assert info.clearance_m < 0.0 and 3.5 < e.data.qpos[0] < 4.3
    assert steps * CFG.sim.ctrl_dt_s < 20.0


def test_reaching_the_goal_ends_in_success():
    e = env(vehicle=nav_eval_vehicle(0))
    e.reset(params=straight_line_episode(e, goal_x_m=3.0, pillar_x_m=None), noise_seed=0)
    done = False
    while not done:
        _, reward, done, info = e.step(np.array([0.6, 0.0, 0.0, 0.0]))
    assert info.success and info.dist_to_goal_m < CFG.reward.goal_radius_m
    assert reward > CFG.reward.success_bonus - 1.0


def test_timeout():
    short = dataclasses.replace(CFG, sim=SimConfig(episode_duration_s=1.0))
    e = env(short, vehicle=nav_eval_vehicle(0))
    e.reset(params=straight_line_episode(e, goal_x_m=20.0, pillar_x_m=None), noise_seed=0)
    steps, done = 0, False
    while not done:
        _, _, done, info = e.step(np.zeros(4))
        steps += 1
    assert info.timeout and not info.success and not info.collision
    assert steps == short.sim.episode_steps
