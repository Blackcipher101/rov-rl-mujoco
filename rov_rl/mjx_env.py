"""MJX (JAX) environment for Brax PPO, numerically mirroring rov_rl.cpu_env.

Per-episode domain randomization: parameters are sampled in reset() and kept in
state.info; step() builds the per-environment model with tree_replace (current
= option.wind, drag = density/viscosity scale, mass/inertia scale, thruster
lag). Train with wrap_for_brax_training(..., full_reset=True) so every episode
re-samples its parameters instead of replaying a cached first state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from rov_rl import core
from rov_rl.config import TRAIN_RANDOMIZATION, RandomizationConfig, RewardConfig, SimConfig
from rov_rl.vehicle import MJCF_PATH, load_model

METRIC_KEYS = ("pos_coarse", "pos_fine", "yaw", "effort", "action_rate", "tilt", "pos_err_m", "thrust_impulse_ns")


class RovMjxEnv(mjx_env.MjxEnv):
    def __init__(
        self,
        randomization: RandomizationConfig = TRAIN_RANDOMIZATION,
        sim: SimConfig = SimConfig(),
        reward: RewardConfig = RewardConfig(),
    ) -> None:
        cfg = config_dict.ConfigDict()
        cfg.ctrl_dt = sim.ctrl_dt_s
        cfg.sim_dt = sim.sim_dt_s
        super().__init__(cfg)
        self._randomization = randomization
        self._sim = sim
        self._reward_cfg = reward
        self._mj_model = load_model()
        if abs(self._mj_model.opt.timestep - sim.sim_dt_s) > 1e-12:
            raise ValueError(f"MJCF timestep {self._mj_model.opt.timestep} != SimConfig.sim_dt_s {sim.sim_dt_s}")
        self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
        self._rov_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "rov")
        self._nominal_mass = jnp.asarray(self._mj_model.body_mass)
        self._nominal_inertia = jnp.asarray(self._mj_model.body_inertia)
        self._nominal_density = float(self._mj_model.opt.density)
        self._nominal_viscosity = float(self._mj_model.opt.viscosity)
        self._nominal_dynprm = jnp.asarray(self._mj_model.actuator_dynprm)

    # --- MjxEnv interface -------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(MJCF_PATH)

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # --- dynamics ---------------------------------------------------------
    def episode_model(self, p: core.EpisodeParams, t_s: jax.Array) -> mjx.Model:
        """Per-episode model; mirrors cpu_env.apply_episode_params + current."""
        return self._mjx_model.tree_replace({
            "opt.wind": core.current_velocity(jnp, t_s, p),
            "opt.density": jnp.asarray(self._nominal_density) * p.drag_scale,
            "opt.viscosity": jnp.asarray(self._nominal_viscosity) * p.drag_scale,
            "body_mass": self._nominal_mass * p.mass_scale,
            "body_inertia": self._nominal_inertia * p.mass_scale,
            "actuator_dynprm": self._nominal_dynprm.at[:, 0].set(p.thruster_tau_s),
        })

    def reset(self, rng: jax.Array) -> mjx_env.State:
        u = jax.random.uniform(rng, (core.NUM_UNIFORMS,))
        p = core.sample_episode(jnp, u, self._randomization, self._sim)
        t0 = jnp.zeros(())
        model = self.episode_model(p, t0)
        data = mjx.make_data(self._mj_model, impl="jax")  # shapes only; same pattern as playground envs
        qpos = data.qpos.at[:3].set(p.start_pos_m).at[3:7].set(p.start_quat)
        data = mjx.forward(model, data.replace(qpos=qpos))
        integral = jnp.zeros(3)
        last_action = jnp.zeros(self.action_size)
        ref = core.reference(jnp, t0, p)
        obs = core.observation(jnp, data.qpos, data.qvel, ref, integral, last_action, self._sim)
        info: dict[str, Any] = {"params": p, "t_s": t0, "integral": integral, "last_action": last_action}
        metrics = {k: jnp.zeros(()) for k in METRIC_KEYS}
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        p: core.EpisodeParams = state.info["params"]
        t_s = state.info["t_s"]
        action = jnp.clip(action, -1.0, 1.0)
        model = self.episode_model(p, t_s)
        ctrl = core.thruster_ctrl(jnp, action, t_s, p)
        sim_dt = self._sim.sim_dt_s

        def substep(data: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
            xfrc = data.xfrc_applied.at[self._rov_id].set(core.buoyancy_wrench(jnp, data.qpos[3:7], p))
            data = mjx.step(model, data.replace(ctrl=ctrl, xfrc_applied=xfrc))
            return data, jnp.sum(jnp.abs(data.actuator_force)) * sim_dt

        data, impulses = jax.lax.scan(substep, state.data, None, length=self._sim.n_substeps)
        t_next = t_s + self._sim.ctrl_dt_s
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        ref = core.reference(jnp, t_next, p)
        err = core.tracking_errors(jnp, data.qpos, ref)
        reward, done, terms = core.reward_and_done(
            jnp, err, action, state.info["last_action"], finite, self._reward_cfg)
        integral = core.update_integral(jnp, state.info["integral"], jnp.nan_to_num(err.pos_err_world_m), self._sim)
        obs = core.observation(jnp, data.qpos, data.qvel, ref, integral, action, self._sim)
        info = {**state.info, "t_s": t_next, "integral": integral, "last_action": action}
        metrics = {
            **{k: jnp.nan_to_num(v) for k, v in terms.items()},
            "pos_err_m": jnp.nan_to_num(err.pos_err_m, nan=self._reward_cfg.max_pos_error_m),
            "thrust_impulse_ns": jnp.sum(impulses),
        }
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)
