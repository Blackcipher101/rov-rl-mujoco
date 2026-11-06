"""Nav / nav2: MJX env vs CPU env on identical episodes and open-loop commands (static and moving obstacles,
fan memory), and the analytic sonar vs mjx.ray.

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
from rov_rl.config import (  # noqa: E402
    NAV2_CONFIG,
    NAV_TRAIN_VEHICLE,
    NavConfig,
    SonarConfig,
    nav_eval_scene,
    nav_eval_scene_moving,
)
from rov_rl.nav_cpu_env import OBSTACLE_GEOM_GROUP, NavCpuEnv, build_scene_model  # noqa: E402
from rov_rl.nav_mjx_env import NavMjxEnv  # noqa: E402

STEPS = 150  # 6 s
POS_TOL_M = 2e-3  # float32 MJX vs float64 CPU
OBS_TOL = 5e-3
MAX_EDGE_BEAMS = 2  # fan beams allowed to sit on a silhouette edge in the open-loop phase
MAX_EDGE_SECTORS = 2  # memory sectors allowed to differ in the open-loop phase (see check_open_loop_obs)


def noiseless(cfg: NavConfig) -> NavConfig:
    """Sensor noise comes from different generators (jax.random vs numpy), so parity is checked noise-free; the
    noise model itself is shared code (nav_core.measure), tested on the CPU side."""
    return dataclasses.replace(cfg, sonar=dataclasses.replace(
        SonarConfig(), range_noise_std_m=0.0, range_noise_frac=0.0, dropout_prob=0.0, dvl_noise_std_mps=0.0))


NOISELESS = noiseless(NavConfig())
SCENE = nav_eval_scene("high")
# (task, config, scene): the nav task, and nav2 with fan memory and moving obstacles.
CASES = {
    "nav": (NOISELESS, SCENE),
    "nav2_moving": (noiseless(NAV2_CONFIG), nav_eval_scene_moving("high")),
}


def to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float64), tree)


def check_open_loop_obs(obs_mjx: np.ndarray, obs_cpu: np.ndarray, qpos_mjx: np.ndarray, scene_now: nav_core.NavScene,
                        cfg: NavConfig, step: int, edge_seen: bool) -> bool:
    """Open-loop observation parity, tolerant of silhouette edges only. Returns whether a fan edge was seen.

    After many open-loop steps the float32 and float64 states differ by up to POS_TOL_M.
    * A fan ray grazing an obstacle silhouette is discontinuous in the pose (it hits a near obstacle or the one
      behind it). Each differing fan beam (at most MAX_EDGE_BEAMS) must be reproduced by the CPU ray code re-cast at
      the MJX pose (scene at the current time, movers included): the ray code agrees, only the state drift differs.
    * Memory sectors (nav2): a remembered point on a sector boundary may land in the neighbouring sector. Each
      differing sector (at most MAX_EDGE_SECTORS) must have the same minimum over itself and its two neighbours in
      both envs, unless a fan edge was seen earlier (the flipped echo may then have been stored in the memory).
    * Every other element must match within OBS_TOL * 4.
    """
    sonar = cfg.sonar
    n_base = nav_core.nav_obs_size(sonar)
    bad = np.flatnonzero(np.abs(obs_mjx - obs_cpu) > OBS_TOL * 4)
    fan = bad[bad < sonar.num_beams]
    mem = bad[bad >= n_base] - n_base
    other = bad[(bad >= sonar.num_beams) & (bad < n_base)]
    assert other.size == 0, f"step {step}: obs elements {other.tolist()} differ"
    assert fan.size <= MAX_EDGE_BEAMS, f"step {step}: fan beams {fan.tolist()} differ"
    if fan.size:
        fan_at_mjx_pose, _ = nav_core.true_ranges(np, qpos_mjx, scene_now, sonar)
        np.testing.assert_allclose(obs_mjx[fan], fan_at_mjx_pose[fan] / sonar.max_range_m, atol=OBS_TOL,
                                   err_msg=f"step {step}: beams {fan.tolist()} not explained by the pose difference")
    if mem.size:
        assert mem.size <= MAX_EDGE_SECTORS, f"step {step}: memory sectors {mem.tolist()} differ"
        if not edge_seen:
            pm, pc = obs_mjx[n_base:], obs_cpu[n_base:]
            n = pm.size
            for k in mem:
                window = [(k - 1) % n, k, (k + 1) % n]
                assert abs(pm[window].min() - pc[window].min()) <= OBS_TOL * 4, (
                    f"step {step}: memory sector {k} differs beyond a boundary crossing")
    return edge_seen or bool(fan.size)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nav_mjx_matches_cpu_open_loop(case, seed):
    cfg, scene_cfg = CASES[case]
    mjx_env = NavMjxEnv(NAV_TRAIN_VEHICLE, scene_cfg, cfg)
    state = jax.jit(mjx_env.reset)(jax.random.PRNGKey(seed))
    p = state.info["params"]
    params_np = nav_core.NavEpisode(vehicle=core.EpisodeParams(*to_numpy(tuple(p.vehicle))),
                                    scene=nav_core.NavScene(*to_numpy(tuple(p.scene))))
    if case == "nav2_moving":
        assert np.any(params_np.scene.mov_active), "the moving case must exercise movers"
    cpu_env = NavCpuEnv(NAV_TRAIN_VEHICLE, scene_cfg, cfg)
    obs_cpu = cpu_env.reset(params=params_np, noise_seed=0)
    assert state.obs.shape == obs_cpu.shape == (nav_core.nav_obs_size(cfg.sonar, cfg.memory),)
    np.testing.assert_allclose(np.asarray(state.obs), obs_cpu, atol=OBS_TOL)

    step = jax.jit(mjx_env.step)
    t = np.arange(STEPS) * cfg.sim.ctrl_dt_s
    cmds = 0.6 * np.sin(np.outer(t, np.array([0.4, 0.9, 0.7, 1.3])) + np.arange(4))
    cmds[:, 0] = 0.5 + 0.4 * np.sin(0.5 * t)  # mostly forward, so the vehicle meets the obstacle field
    edge_seen = False
    for k in range(STEPS):
        state = step(state, jnp.asarray(cmds[k], dtype=jnp.float32))
        obs_cpu, r_cpu, done_cpu, info = cpu_env.step(cmds[k])
        cpu_terminated = float(info.success or info.collision or info.failure)
        if float(state.done) != cpu_terminated:
            # float32 vs float64 may put a termination threshold crossing one step apart.
            near_threshold = (abs(info.clearance_m) < 2e-3
                              or abs(info.dist_to_goal_m - cfg.reward.goal_radius_m) < 2e-3)
            assert near_threshold, f"termination differs at step {k} away from any threshold: {info}"
            break
        np.testing.assert_allclose(np.asarray(state.data.qpos[:3]), cpu_env.data.qpos[:3], atol=POS_TOL_M,
                                   err_msg=f"position diverged at step {k}")
        assert float(state.reward) == pytest.approx(r_cpu, abs=5e-3)
        edge_seen = check_open_loop_obs(np.asarray(state.obs, dtype=np.float64), obs_cpu,
                                        np.asarray(state.data.qpos, np.float64), cpu_env.scene_now(), cfg, k,
                                        edge_seen)
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
