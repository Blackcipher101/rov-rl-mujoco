"""Nav task: MJX env vs CPU env on identical episodes and open-loop commands, and the analytic sonar vs mjx.ray.

Skipped when JAX / MJX / playground are not installed (e.g. CPU-only laptop); run on the GPU workstation as
part of scripts/run_ws1.sh.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
mjx = pytest.importorskip("mujoco.mjx")
pytest.importorskip("mujoco_playground")

import mujoco  # noqa: E402

from rov_rl import core, nav_core  # noqa: E402
from rov_rl.config import NAV_TRAIN_VEHICLE, NavConfig, SonarConfig, nav_eval_scene  # noqa: E402
from rov_rl.nav_cpu_env import OBSTACLE_GEOM_GROUP, NavCpuEnv, build_scene_model  # noqa: E402
from rov_rl.nav_mjx_env import NavMjxEnv  # noqa: E402

STEPS = 150  # 6 s
POS_TOL_M = 2e-3  # float32 MJX vs float64 CPU
OBS_TOL = 5e-3
MAX_EDGE_BEAMS = 2  # fan beams allowed to sit on a silhouette edge in the open-loop phase
# Sensor noise comes from different generators (jax.random vs numpy), so parity is checked noise-free; the
# noise model itself is shared code (nav_core.measure), tested on the CPU side.
NOISELESS = dataclasses.replace(NavConfig(), sonar=dataclasses.replace(
    SonarConfig(), range_noise_std_m=0.0, range_noise_frac=0.0, dropout_prob=0.0, dvl_noise_std_mps=0.0))
SCENE = nav_eval_scene("high")


def to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float64), tree)


def check_open_loop_obs(obs_mjx: np.ndarray, obs_cpu: np.ndarray, qpos_mjx: np.ndarray,
                        scene: nav_core.NavScene, step: int) -> None:
    """Open-loop observation parity, tolerant of silhouette edges only.

    After many open-loop steps the float32 and float64 states differ by up to POS_TOL_M. A fan ray grazing an
    obstacle silhouette is discontinuous in the pose (it hits a near obstacle or the one behind it), so such a
    beam may legitimately differ. Each differing element must be a fan beam (at most MAX_EDGE_BEAMS of them), and
    the CPU ray code re-cast at the MJX pose must reproduce the MJX value: then the ray code agrees and only the
    state drift differs. Every other element must match within OBS_TOL * 4.
    """
    sonar = NOISELESS.sonar
    bad = np.flatnonzero(np.abs(obs_mjx - obs_cpu) > OBS_TOL * 4)
    if bad.size == 0:
        return
    assert np.all(bad < sonar.num_beams) and bad.size <= MAX_EDGE_BEAMS, (
        f"step {step}: obs elements {bad.tolist()} differ (mjx {obs_mjx[bad]}, cpu {obs_cpu[bad]})")
    fan_at_mjx_pose, _ = nav_core.true_ranges(np, qpos_mjx, scene, sonar)
    np.testing.assert_allclose(obs_mjx[bad], fan_at_mjx_pose[bad] / sonar.max_range_m, atol=OBS_TOL,
                               err_msg=f"step {step}: beams {bad.tolist()} not explained by the pose difference")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nav_mjx_matches_cpu_open_loop(seed):
    mjx_env = NavMjxEnv(NAV_TRAIN_VEHICLE, SCENE, NOISELESS)
    state = jax.jit(mjx_env.reset)(jax.random.PRNGKey(seed))
    p = state.info["params"]
    params_np = nav_core.NavEpisode(vehicle=core.EpisodeParams(*to_numpy(tuple(p.vehicle))),
                                    scene=nav_core.NavScene(*to_numpy(tuple(p.scene))))
    cpu_env = NavCpuEnv(NAV_TRAIN_VEHICLE, SCENE, NOISELESS)
    obs_cpu = cpu_env.reset(params=params_np, noise_seed=0)
    assert state.obs.shape == obs_cpu.shape == (nav_core.nav_obs_size(NOISELESS.sonar),)
    np.testing.assert_allclose(np.asarray(state.obs), obs_cpu, atol=OBS_TOL)

    step = jax.jit(mjx_env.step)
    t = np.arange(STEPS) * NOISELESS.sim.ctrl_dt_s
    cmds = 0.6 * np.sin(np.outer(t, np.array([0.4, 0.9, 0.7, 1.3])) + np.arange(4))
    cmds[:, 0] = 0.5 + 0.4 * np.sin(0.5 * t)  # mostly forward, so the vehicle meets the obstacle field
    for k in range(STEPS):
        state = step(state, jnp.asarray(cmds[k], dtype=jnp.float32))
        obs_cpu, r_cpu, done_cpu, info = cpu_env.step(cmds[k])
        cpu_terminated = float(info.success or info.collision or info.failure)
        if float(state.done) != cpu_terminated:
            # float32 vs float64 may put a termination threshold crossing one step apart.
            near_threshold = (abs(info.clearance_m) < 2e-3
                              or abs(info.dist_to_goal_m - NOISELESS.reward.goal_radius_m) < 2e-3)
            assert near_threshold, f"termination differs at step {k} away from any threshold: {info}"
            break
        np.testing.assert_allclose(np.asarray(state.data.qpos[:3]), cpu_env.data.qpos[:3], atol=POS_TOL_M,
                                   err_msg=f"position diverged at step {k}")
        assert float(state.reward) == pytest.approx(r_cpu, abs=5e-3)
        check_open_loop_obs(np.asarray(state.obs, dtype=np.float64), obs_cpu, np.asarray(state.data.qpos, np.float64),
                            params_np.scene, k)
        if done_cpu:
            break


def test_nav_metrics_keep_wrapper_keys():
    """step() must start from state.metrics (brax's EvalWrapper adds 'reward')."""
    env = NavMjxEnv(NAV_TRAIN_VEHICLE, SCENE, NOISELESS)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = state.replace(metrics={**state.metrics, "reward": jnp.zeros(())})
    out = jax.jit(env.step)(state, jnp.zeros(4))
    assert set(out.metrics) == set(state.metrics)


@pytest.mark.parametrize("seed", [0, 1])
def test_analytic_sonar_matches_mjx_ray(seed):
    """The analytic caster agrees with MJX's own ray caster (capsule, box, plane) on the compiled scene."""
    rng = np.random.default_rng(seed)
    scene = nav_core.sample_scene(np, rng.random(nav_core.NAV_NUM_UNIFORMS), SCENE)
    model = build_scene_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    m = mjx.put_model(model, impl="jax")
    d = mjx.put_data(model, data, impl="jax")
    group = tuple(int(g == OBSTACLE_GEOM_GROUP) for g in range(mujoco.mjNGROUP))
    ray = jax.jit(lambda pnt, vec: mjx.ray(m, d, pnt, vec, geomgroup=group))
    max_range = 40.0
    goal = np.asarray(scene.goal_m)
    hits = 0
    for _ in range(100):
        xy = rng.uniform(-0.1, 1.1) * goal[:2] + rng.uniform(-6.0, 6.0, size=2)
        o = np.array([xy[0], xy[1], rng.uniform(float(scene.seabed_z_m) + 0.1, 3.0)])
        if nav_core.scene_distance(np, o, scene) < 0.05:
            continue
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        ref, _ = ray(jnp.asarray(o, dtype=jnp.float32), jnp.asarray(v, dtype=jnp.float32))
        ref = float(ref)
        ref = max_range if ref < 0.0 else min(ref, max_range)
        ours = min(float(nav_core.ray_scene(np, o, v[None], scene)[0]), max_range)
        hits += ref < max_range
        assert ours == pytest.approx(ref, abs=5e-3)  # float32 MJX
    assert hits > 10
