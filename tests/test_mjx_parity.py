"""MJX env vs CPU env on identical episodes and open-loop actions.

Skipped when JAX / MJX / playground are not installed (e.g. CPU-only laptop);
run on the GPU workstation as part of scripts/run_ws1.sh.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("mujoco.mjx")
pytest.importorskip("mujoco_playground")

from rov_rl import core  # noqa: E402
from rov_rl.config import TRAIN_RANDOMIZATION, eval_randomization  # noqa: E402
from rov_rl.cpu_env import RovCpuEnv  # noqa: E402
from rov_rl.mjx_env import RovMjxEnv  # noqa: E402

STEPS = 150  # 6 s
POS_TOL_M = 2e-3  # float32 MJX vs float64 CPU
OBS_TOL = 5e-3
# Thruster failure switches on a time threshold; float32 vs float64 time can put
# the switch one control step apart, so parity is checked without failures
# (the failure logic itself is shared code, tested in test_core.py).
PARITY_RANDOMIZATION = dataclasses.replace(TRAIN_RANDOMIZATION, p_thruster_failure=0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mjx_matches_cpu_open_loop(seed):
    mjx_env = RovMjxEnv(PARITY_RANDOMIZATION)
    state = jax.jit(mjx_env.reset)(jax.random.PRNGKey(seed))
    params_np = core.EpisodeParams(*[np.asarray(x, dtype=np.float64) for x in state.info["params"]])
    cpu_env = RovCpuEnv(PARITY_RANDOMIZATION)
    obs_cpu = cpu_env.reset(params=params_np)
    np.testing.assert_allclose(np.asarray(state.obs), obs_cpu, atol=OBS_TOL)

    step = jax.jit(mjx_env.step)
    t = np.arange(STEPS) * cpu_env.sim.ctrl_dt_s
    actions = 0.4 * np.sin(np.outer(t, np.arange(1, 9) * 0.7) + np.arange(8))
    for k in range(STEPS):
        state = step(state, jnp.asarray(actions[k], dtype=jnp.float32))
        obs_cpu, r_cpu, _, info = cpu_env.step(actions[k])
        if info.terminated:
            break
        np.testing.assert_allclose(np.asarray(state.data.qpos[:3]), cpu_env.data.qpos[:3], atol=POS_TOL_M,
                                   err_msg=f"position diverged at step {k}")
        assert float(state.reward) == pytest.approx(r_cpu, abs=5e-3)
    np.testing.assert_allclose(np.asarray(state.obs), obs_cpu, atol=OBS_TOL * 4)


def test_mjx_reset_samples_eval_config():
    env = RovMjxEnv(eval_randomization("station", 4))
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert float(jnp.linalg.norm(state.info["params"].current_const_mps)) == pytest.approx(0.4, rel=1e-5)
    assert state.obs.shape == (core.OBS_SIZE,)
