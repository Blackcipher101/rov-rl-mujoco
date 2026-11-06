"""CPU environment behaviour and PID baseline regression."""

from __future__ import annotations

import numpy as np
import pytest

from rov_rl import core
from rov_rl.config import MetricConfig, RandomizationConfig, TRAIN_RANDOMIZATION, eval_randomization
from rov_rl.cpu_env import RovCpuEnv
from rov_rl.pid_baseline import CascadePid, PidGains
from rov_rl.rollout import PidController, episode_metrics, run_episode, settling_time_s


def test_reset_is_deterministic_and_obs_valid():
    env = RovCpuEnv(TRAIN_RANDOMIZATION)
    a, b = env.reset(seed=11), env.reset(seed=11)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (core.OBS_SIZE,)
    obs, reward, done, info = env.step(np.zeros(8))
    assert obs.shape == (core.OBS_SIZE,) and np.isfinite(reward) and not done
    assert info.t_s == pytest.approx(env.sim.ctrl_dt_s)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        RovCpuEnv(TRAIN_RANDOMIZATION).step(np.zeros(8))


def test_episode_truncates_at_duration():
    env = RovCpuEnv(eval_randomization("station", 0))
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, done, info = env.step(np.zeros(8))
        steps += 1
    assert steps == env.sim.episode_steps and not info.terminated


def test_randomized_params_are_restored_on_next_reset():
    env = RovCpuEnv(TRAIN_RANDOMIZATION)
    nominal_mass = env.model.body_mass.copy()
    nominal_density = env.model.opt.density
    p = env.sample_params(np.random.default_rng(0))._replace(mass_scale=np.float64(1.1),
                                                             drag_scale=np.float64(1.3))
    env.reset(params=p)
    np.testing.assert_allclose(env.model.body_mass, nominal_mass * 1.1)
    assert env.model.opt.density == pytest.approx(nominal_density * 1.3)
    env.reset(params=p._replace(mass_scale=np.float64(1.0), drag_scale=np.float64(1.0)))
    np.testing.assert_allclose(env.model.body_mass, nominal_mass)
    assert env.model.opt.density == pytest.approx(nominal_density)


def test_current_drifts_uncontrolled_vehicle_downstream():
    cfg = RandomizationConfig(p_station=1.0, current_const_mps=(0.2, 0.2), current_amp_mps=(0.0, 0.0),
                              station_start_pos_m=0.0, start_tilt_rad=0.0, current_elevation_max_rad=0.0,
                              buoyancy_ratio=(1.0, 1.0))
    env = RovCpuEnv(cfg)
    env.reset(seed=1)
    for _ in range(250):  # 10 s
        env.step(np.zeros(8))
    drift = env.data.qpos[:3]
    direction = env.params.current_const_mps / np.linalg.norm(env.params.current_const_mps)
    assert float(drift @ direction) > 1.0  # carried > 1 m downstream within 10 s
    assert np.linalg.norm(env.data.qvel[:3] - env.params.current_const_mps) < 0.02  # moves with the water


def test_settling_time_definition():
    cfg = MetricConfig()
    t = np.arange(1, 11, dtype=float)
    err = np.array([0.5, 0.3, 0.05, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
    assert settling_time_s(t, err, np.zeros(10), cfg) == 4.0
    assert np.isnan(settling_time_s(t, err[::-1], np.zeros(10), cfg))


@pytest.fixture(scope="module")
def pid_gains() -> PidGains:
    return PidGains.load()


@pytest.mark.parametrize("task,level,max_rms_m", [("station", 0, 0.05), ("station", 2, 0.1), ("track", 1, 0.15)])
def test_pid_baseline_regression(pid_gains, task, level, max_rms_m):
    env = RovCpuEnv(eval_randomization(task, level))
    ctrl = PidController(CascadePid(env.model, pid_gains, env.sim.ctrl_dt_s))
    log = run_episode(env, ctrl, env.sample_params(np.random.default_rng(123)))
    m = episode_metrics(log, MetricConfig(), env.sim.episode_duration_s)
    assert not m["terminated"]
    assert m["pos_rms_m"] < max_rms_m
    assert m["success"]
