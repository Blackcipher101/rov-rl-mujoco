"""MJX (JAX) environment of the nav task for Brax PPO, numerically mirroring rov_rl.nav_cpu_env.

Reuses RovMjxEnv for the vehicle model and the per-episode plant / current model (episode_model). Per
episode, reset() samples the vehicle parameters and an obstacle field (rov_rl/nav_core.py) and keeps them in
state.info; obstacles are analytic primitives and never enter the MJX physics model. Train with
wrap_for_brax_training(..., full_reset=True) so every episode gets a new field.

Action: normalized velocity command (surge, sway, heave, yaw rate) in [-1, 1]^4, converted to thruster
commands by the shared velocity controller (nav_core.command_to_thrust), as for the classical baseline.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env

from rov_rl import core, nav_core
from rov_rl.config import (NAV_TRAIN_SCENE, NAV_TRAIN_VEHICLE, NavConfig, NavSceneConfig, RandomizationConfig,
                           RewardConfig)
from rov_rl.mjx_env import RovMjxEnv
from rov_rl.nav_cpu_env import make_velocity_controller_params
from rov_rl.pid_baseline import PidGains

NAV_METRIC_KEYS = ("progress", "clearance", "effort", "action_rate", "time", "tilt",
                   "success", "collision", "failure", "thrust_impulse_ns")


class NavMjxEnv(RovMjxEnv):
    def __init__(
        self,
        vehicle: RandomizationConfig = NAV_TRAIN_VEHICLE,
        scene: NavSceneConfig = NAV_TRAIN_SCENE,
        cfg: NavConfig = NavConfig(),
        gains: PidGains | None = None,
    ) -> None:
        # RovMjxEnv's tracking reward is not used by this env.
        super().__init__(vehicle, cfg.sim, RewardConfig())
        self._scene_cfg = scene
        self._nav_cfg = cfg
        vc = make_velocity_controller_params(self._mj_model, gains if gains is not None else PidGains.load())
        self._vc = vc._replace(
            tam_pinv=jnp.asarray(vc.tam_pinv, dtype=jnp.float32),
            inertia_kgm2=jnp.asarray(vc.inertia_kgm2, dtype=jnp.float32),
            lin_damp=jnp.asarray(vc.lin_damp, dtype=jnp.float32),
            quad_damp=jnp.asarray(vc.quad_damp, dtype=jnp.float32),
        )

    @property
    def action_size(self) -> int:
        return 4

    def _observe(self, data: mjx.Data, scene: nav_core.NavScene, last_cmd: jax.Array,
                 rng: jax.Array) -> jax.Array:
        sonar = self._nav_cfg.sonar
        k_n, k_u = jax.random.split(rng)
        normal = jax.random.normal(k_n, (nav_core.num_noise_normals(sonar),))
        uniform = jax.random.uniform(k_u, (nav_core.num_noise_uniforms(sonar),))
        meas = nav_core.measure(jnp, data.qpos, data.qvel, scene, sonar, self._scene_cfg.surface_z_m, normal,
                                uniform)
        return nav_core.nav_observation(jnp, meas, last_cmd, sonar, self._scene_cfg.surface_z_m)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        k_v, k_s, k_obs, k_next = jax.random.split(rng, 4)
        u_v = jax.random.uniform(k_v, (core.NUM_UNIFORMS,))
        u_s = jax.random.uniform(k_s, (nav_core.NAV_NUM_UNIFORMS,))
        p = nav_core.make_episode(jnp, u_v, u_s, self._randomization, self._scene_cfg, self._sim)
        t0 = jnp.zeros(())
        model = self.episode_model(p.vehicle, t0)
        data = mjx.make_data(self._mj_model, impl="jax")
        qpos = data.qpos.at[:3].set(p.vehicle.start_pos_m).at[3:7].set(p.vehicle.start_quat)
        data = mjx.forward(model, data.replace(qpos=qpos))
        last_cmd = jnp.zeros(self.action_size)
        obs = self._observe(data, p.scene, last_cmd, k_obs)
        dist = jnp.sqrt(jnp.sum((p.scene.goal_m - data.qpos[:3]) ** 2))
        info: dict[str, Any] = {"params": p, "t_s": t0, "last_cmd": last_cmd, "dist_m": dist, "rng": k_next}
        metrics = {k: jnp.zeros(()) for k in NAV_METRIC_KEYS}
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        p: nav_core.NavEpisode = state.info["params"]
        t_s = state.info["t_s"]
        cmd = jnp.clip(action, -1.0, 1.0)
        thrust_action = nav_core.command_to_thrust(jnp, state.data.qpos, state.data.qvel, cmd,
                                                   self._nav_cfg.command, self._vc)
        model = self.episode_model(p.vehicle, t_s)
        ctrl = core.thruster_ctrl(jnp, thrust_action, t_s, p.vehicle)
        sim_dt = self._sim.sim_dt_s

        def substep(data: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
            xfrc = data.xfrc_applied.at[self._rov_id].set(core.buoyancy_wrench(jnp, data.qpos[3:7], p.vehicle))
            data = mjx.step(model, data.replace(ctrl=ctrl, xfrc_applied=xfrc))
            return data, jnp.sum(jnp.abs(data.actuator_force)) * sim_dt

        data, impulses = jax.lax.scan(substep, state.data, None, length=self._sim.n_substeps)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        dist = jnp.sqrt(jnp.sum((p.scene.goal_m - data.qpos[:3]) ** 2))
        clearance = nav_core.clearance_m(jnp, data.qpos[:3], p.scene)
        cos_tilt = core.quat_to_mat(jnp, data.qpos[3:7])[2, 2]
        reward, outcome, terms = nav_core.nav_reward_and_done(
            jnp, state.info["dist_m"], dist, clearance, cos_tilt, thrust_action, cmd, state.info["last_cmd"],
            finite, self._nav_cfg.reward)
        k_obs, k_next = jax.random.split(state.info["rng"])
        obs = self._observe(data, p.scene, cmd, k_obs)
        info = {**state.info, "t_s": t_s + self._sim.ctrl_dt_s, "last_cmd": cmd, "dist_m": dist, "rng": k_next}
        # Start from the incoming metrics: brax wrappers add keys (e.g. 'reward' in EvalWrapper) and
        # jax.lax.scan needs the same pytree structure on every step.
        metrics = {
            **state.metrics,
            **{k: jnp.nan_to_num(v) for k, v in terms.items()},
            "success": outcome.success,
            "collision": outcome.collision,
            "failure": outcome.failure,
            "thrust_impulse_ns": jnp.sum(impulses),
        }
        return state.replace(data=data, obs=obs, reward=reward, done=outcome.done, metrics=metrics, info=info)
