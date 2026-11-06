"""NumPy / CPU-MuJoCo environment, numerically mirroring rov_rl.mjx_env.

Used for unit tests, the PID baseline, evaluation of both controllers under
identical conditions, and video rendering. No JAX dependency.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np

from rov_rl import core
from rov_rl.config import RandomizationConfig, RewardConfig, SimConfig
from rov_rl.vehicle import load_model


@dataclasses.dataclass
class StepInfo:
    t_s: float
    pos_err_m: float
    yaw_err_rad: float
    thrust_impulse_ns: float  # sum over thrusters of |F| dt during this control step
    terminated: bool
    reward_terms: dict[str, float]


class NominalModelState:
    """Snapshot of model fields the randomization overwrites, for exact restore."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.body_mass = model.body_mass.copy()
        self.body_inertia = model.body_inertia.copy()
        self.density = float(model.opt.density)
        self.viscosity = float(model.opt.viscosity)
        self.dynprm = model.actuator_dynprm.copy()


def apply_episode_params(model: mujoco.MjModel, nominal: NominalModelState, p: core.EpisodeParams) -> None:
    """Write per-episode plant parameters into a CPU MjModel (mirrors the MJX env).

    Mass and inertia are scaled together, which leaves MuJoCo's inertia-box
    drag unchanged; drag is varied independently through density/viscosity.
    """
    model.body_mass[:] = nominal.body_mass * float(p.mass_scale)
    model.body_inertia[:] = nominal.body_inertia * float(p.mass_scale)
    model.opt.density = nominal.density * float(p.drag_scale)
    model.opt.viscosity = nominal.viscosity * float(p.drag_scale)
    model.actuator_dynprm[:] = nominal.dynprm
    model.actuator_dynprm[:, 0] = float(p.thruster_tau_s)


class RovCpuEnv:
    """Gym-like single environment: reset(seed) -> obs, step(action) -> (obs, reward, done, info)."""

    def __init__(
        self,
        randomization: RandomizationConfig,
        sim: SimConfig = SimConfig(),
        reward: RewardConfig = RewardConfig(),
    ) -> None:
        self.model = load_model()
        if abs(self.model.opt.timestep - sim.sim_dt_s) > 1e-12:
            raise ValueError(f"MJCF timestep {self.model.opt.timestep} != SimConfig.sim_dt_s {sim.sim_dt_s}")
        self.data = mujoco.MjData(self.model)
        self.randomization = randomization
        self.sim = sim
        self.reward_cfg = reward
        self._nominal = NominalModelState(self.model)
        self._rov_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rov")
        self.params: core.EpisodeParams | None = None
        self.t_s = 0.0
        self.step_count = 0
        self._integral = np.zeros(3)
        self._last_action = np.zeros(self.model.nu)

    def sample_params(self, rng: np.random.Generator) -> core.EpisodeParams:
        return core.sample_episode(np, rng.random(core.NUM_UNIFORMS), self.randomization, self.sim)

    def reset(self, seed: int | None = None, params: core.EpisodeParams | None = None) -> np.ndarray:
        """Start an episode from explicit params, or sample them from `seed`."""
        if params is None:
            params = self.sample_params(np.random.default_rng(seed))
        self.params = params
        apply_episode_params(self.model, self._nominal, params)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = params.start_pos_m
        self.data.qpos[3:7] = params.start_quat
        self.t_s = 0.0
        self.step_count = 0
        self._integral = np.zeros(3)
        self._last_action = np.zeros(self.model.nu)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(core.reference(np, self.t_s, params))

    def reference(self) -> core.Reference:
        self._require_reset()
        return core.reference(np, self.t_s, self.params)

    def _require_reset(self) -> None:
        if self.params is None:
            raise RuntimeError("call reset() before step()")

    def _obs(self, ref: core.Reference) -> np.ndarray:
        return core.observation(np, self.data.qpos, self.data.qvel, ref, self._integral, self._last_action, self.sim)

    @property
    def done_by_time(self) -> bool:
        return self.step_count >= self.sim.episode_steps

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        self._require_reset()
        p = self.params
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.model.opt.wind[:] = core.current_velocity(np, self.t_s, p)
        self.data.ctrl[:] = core.thruster_ctrl(np, action, self.t_s, p)
        impulse = 0.0
        for _ in range(self.sim.n_substeps):
            self.data.xfrc_applied[self._rov_id] = core.buoyancy_wrench(np, self.data.qpos[3:7], p)
            mujoco.mj_step(self.model, self.data)
            impulse += float(np.sum(np.abs(self.data.actuator_force))) * self.sim.sim_dt_s
        self.t_s += self.sim.ctrl_dt_s
        self.step_count += 1

        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        ref = core.reference(np, self.t_s, p)
        err = core.tracking_errors(np, self.data.qpos, ref)
        reward, done, terms = core.reward_and_done(
            np, err, action, self._last_action, np.bool_(finite), self.reward_cfg)
        self._integral = core.update_integral(np, self._integral, np.nan_to_num(err.pos_err_world_m), self.sim)
        self._last_action = action
        info = StepInfo(
            t_s=self.t_s,
            pos_err_m=float(err.pos_err_m),
            yaw_err_rad=float(err.yaw_err_rad),
            thrust_impulse_ns=impulse,
            terminated=bool(done),
            reward_terms={k: float(v) for k, v in terms.items()},
        )
        return self._obs(ref), float(reward), bool(done) or self.done_by_time, info

    def state_snapshot(self) -> dict[str, Any]:
        """Pose/velocity for controllers that use state feedback (PID)."""
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "t_s": self.t_s}
