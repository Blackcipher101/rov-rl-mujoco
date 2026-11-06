"""NumPy / CPU-MuJoCo environment of the nav task, numerically mirroring rov_rl.nav_mjx_env.

Used for tests, the classical baseline, evaluation of both planners on identical episodes, and video. The
physics model is the unchanged vehicle MJCF; obstacles exist only as analytic primitives (rov_rl/nav_core.py).
build_scene_model() compiles them into MuJoCo geoms for rendering and for checking the analytic sonar against
mujoco.mj_ray; those geoms never enter the simulated physics.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from rov_rl import core, nav_core
from rov_rl.config import NavConfig, NavSceneConfig, RandomizationConfig
from rov_rl.cpu_env import NominalModelState, apply_episode_params
from rov_rl.pid_baseline import PidGains
from rov_rl.vehicle import DEFAULT_BUOYANCY_RATIO, GRAVITY_MPS2, MJCF_PATH, VEHICLE, allocation_matrix, load_model

OBSTACLE_GEOM_GROUP = 0  # vehicle geoms are group 1 (MJCF default), so rays can select obstacles only
SEABED_PLANE_HALF_SIZE_M = 100.0
NAV_FOG_END_M = 25.0  # visual only: nav videos look further than the 14 m of the control-task videos
PILLAR_RGBA = (0.42, 0.45, 0.40, 1.0)
PIPE_RGBA = (0.55, 0.36, 0.20, 1.0)
BOX_RGBA = (0.36, 0.33, 0.27, 1.0)


def make_velocity_controller_params(model: mujoco.MjModel, gains: PidGains) -> nav_core.VelocityControllerParams:
    """Velocity-controller constants from the compiled model's allocation matrix and the tuned PID gains."""
    tam = allocation_matrix(model)
    if np.linalg.matrix_rank(tam) < 6:
        raise ValueError("thruster allocation matrix is rank deficient")
    return nav_core.VelocityControllerParams(
        tam_pinv=np.linalg.pinv(tam),
        mass_kg=VEHICLE.mass_kg,
        inertia_kgm2=np.array(VEHICLE.inertia_kgm2),
        lin_damp=np.array(VEHICLE.linear_damping[:3]),
        quad_damp=np.array(VEHICLE.quadratic_damping[:3]),
        net_buoyancy_n=(DEFAULT_BUOYANCY_RATIO - 1.0) * VEHICLE.mass_kg * GRAVITY_MPS2,
        thrust_max_n=VEHICLE.thrust_max_n,
        kp_vel_ps=gains.kp_vel_ps,
        kp_yaw_rate_ps=gains.kp_yaw_rate_ps,
        kp_att_ps2=gains.kp_att_ps2,
        kd_att_ps=gains.kd_att_ps,
    )


def build_scene_model(scene: nav_core.NavScene) -> mujoco.MjModel:
    """Vehicle MJCF plus the scene's active obstacles as visual geoms (group 0) and the seabed moved to depth."""
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.visual.map.fogend = NAV_FOG_END_M
    seabed = spec.geom("seabed")
    seabed.pos = [0.0, 0.0, float(scene.seabed_z_m)]
    seabed.size = [SEABED_PLANE_HALF_SIZE_M, SEABED_PLANE_HALF_SIZE_M, 0.1]
    world = spec.worldbody
    for i in np.flatnonzero(np.asarray(scene.cap_active) > 0.5):
        g = world.add_geom()
        g.name = f"capsule{i}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.fromto = np.concatenate([scene.cap_a_m[i], scene.cap_b_m[i]]).astype(float)
        g.size = [float(scene.cap_radius_m[i]), 0.0, 0.0]
        g.rgba = PIPE_RGBA if scene.cap_is_pipe[i] > 0.5 else PILLAR_RGBA
        g.group = OBSTACLE_GEOM_GROUP
        g.contype = g.conaffinity = 0
    for i in np.flatnonzero(np.asarray(scene.box_active) > 0.5):
        g = world.add_geom()
        g.name = f"box{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = np.asarray(scene.box_center_m[i], dtype=float)
        g.quat = np.asarray(core.yaw_quat(np, 0.0, 0.0, float(scene.box_yaw_rad[i])), dtype=float)
        g.size = np.asarray(scene.box_half_m[i], dtype=float)
        g.rgba = BOX_RGBA
        g.group = OBSTACLE_GEOM_GROUP
        g.contype = g.conaffinity = 0
    return spec.compile()


@dataclasses.dataclass
class NavStepInfo:
    t_s: float
    dist_to_goal_m: float
    clearance_m: float
    thrust_impulse_ns: float  # sum over thrusters of |F| dt during this control step
    success: bool
    collision: bool
    failure: bool  # tilt or non-finite state
    timeout: bool
    reward_terms: dict[str, float]


class NavCpuEnv:
    """reset(seed | params) -> obs; step(cmd) -> (obs, reward, done, info). `measurement` holds the last reading."""

    def __init__(
        self,
        vehicle: RandomizationConfig,
        scene: NavSceneConfig,
        cfg: NavConfig = NavConfig(),
        gains: PidGains | None = None,
    ) -> None:
        self.model = load_model()
        if abs(self.model.opt.timestep - cfg.sim.sim_dt_s) > 1e-12:
            raise ValueError(f"MJCF timestep {self.model.opt.timestep} != SimConfig.sim_dt_s {cfg.sim.sim_dt_s}")
        self.data = mujoco.MjData(self.model)
        self.vehicle_cfg = vehicle
        self.scene_cfg = scene
        self.cfg = cfg
        self.vc = make_velocity_controller_params(self.model, gains if gains is not None else PidGains.load())
        self._nominal = NominalModelState(self.model)
        self._rov_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rov")
        self.params: nav_core.NavEpisode | None = None
        self.measurement: nav_core.NavMeasurement | None = None
        self.t_s = 0.0
        self.step_count = 0
        self._last_cmd = np.zeros(4)
        self._dist_m = 0.0
        self._noise_rng = np.random.default_rng(0)

    @property
    def observation_size(self) -> int:
        return nav_core.nav_obs_size(self.cfg.sonar)

    def sample_params(self, rng: np.random.Generator) -> nav_core.NavEpisode:
        return nav_core.make_episode(np, rng.random(core.NUM_UNIFORMS), rng.random(nav_core.NAV_NUM_UNIFORMS),
                                     self.vehicle_cfg, self.scene_cfg, self.cfg.sim)

    def reset(self, seed: int | None = None, params: nav_core.NavEpisode | None = None,
              noise_seed: int | None = None) -> np.ndarray:
        """Start an episode from explicit params, or sample them from `seed`. Sensor noise uses `noise_seed`
        (default: `seed`), so two controllers given the same seeds see the same noise sequence."""
        if params is None:
            params = self.sample_params(np.random.default_rng(seed))
        self.params = params
        self._noise_rng = np.random.default_rng(noise_seed if noise_seed is not None else seed)
        v = params.vehicle
        apply_episode_params(self.model, self._nominal, v)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = v.start_pos_m
        self.data.qpos[3:7] = v.start_quat
        mujoco.mj_forward(self.model, self.data)
        self.t_s = 0.0
        self.step_count = 0
        self._last_cmd = np.zeros(4)
        self._dist_m = float(np.linalg.norm(params.scene.goal_m - self.data.qpos[:3]))
        return self._observe()

    def _observe(self) -> np.ndarray:
        sonar = self.cfg.sonar
        normal = self._noise_rng.standard_normal(nav_core.num_noise_normals(sonar))
        uniform = self._noise_rng.random(nav_core.num_noise_uniforms(sonar))
        self.measurement = nav_core.measure(np, self.data.qpos, self.data.qvel, self.params.scene, sonar,
                                            self.scene_cfg.surface_z_m, normal, uniform)
        return nav_core.nav_observation(np, self.measurement, self._last_cmd, sonar, self.scene_cfg.surface_z_m)

    def _require_reset(self) -> None:
        if self.params is None:
            raise RuntimeError("call reset() before step()")

    def clearance_m(self) -> float:
        self._require_reset()
        return float(nav_core.clearance_m(np, self.data.qpos[:3], self.params.scene))

    def step(self, cmd: np.ndarray) -> tuple[np.ndarray, float, bool, NavStepInfo]:
        self._require_reset()
        p = self.params
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), -1.0, 1.0)
        thrust_action = nav_core.command_to_thrust(np, self.data.qpos, self.data.qvel, cmd, self.cfg.command, self.vc)
        self.model.opt.wind[:] = core.current_velocity(np, self.t_s, p.vehicle)
        self.data.ctrl[:] = core.thruster_ctrl(np, thrust_action, self.t_s, p.vehicle)
        impulse = 0.0
        for _ in range(self.cfg.sim.n_substeps):
            self.data.xfrc_applied[self._rov_id] = core.buoyancy_wrench(np, self.data.qpos[3:7], p.vehicle)
            mujoco.mj_step(self.model, self.data)
            impulse += float(np.sum(np.abs(self.data.actuator_force))) * self.cfg.sim.sim_dt_s
        self.t_s += self.cfg.sim.ctrl_dt_s
        self.step_count += 1

        qpos = self.data.qpos
        finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(self.data.qvel)))
        dist = float(np.linalg.norm(p.scene.goal_m - qpos[:3]))
        clearance = float(nav_core.clearance_m(np, qpos[:3], p.scene))
        cos_tilt = core.quat_to_mat(np, qpos[3:7])[2, 2]
        reward, outcome, terms = nav_core.nav_reward_and_done(
            np, np.float64(self._dist_m), np.float64(dist), np.float64(clearance), cos_tilt, thrust_action, cmd,
            self._last_cmd, np.bool_(finite), self.cfg.reward)
        self._dist_m = dist
        self._last_cmd = cmd
        obs = self._observe()
        terminated = bool(outcome.done)
        timeout = (not terminated) and self.step_count >= self.cfg.sim.episode_steps
        info = NavStepInfo(
            t_s=self.t_s,
            dist_to_goal_m=dist,
            clearance_m=clearance,
            thrust_impulse_ns=impulse,
            success=bool(outcome.success),
            collision=bool(outcome.collision),
            failure=bool(outcome.failure),
            timeout=timeout,
            reward_terms={k: float(v) for k, v in terms.items()},
        )
        return obs, float(reward), terminated or timeout, info
