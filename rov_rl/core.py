"""Array-backend-agnostic environment logic shared by the CPU and MJX envs.

Every function takes ``xp`` (``numpy`` or ``jax.numpy``) and uses only
operations available in both, with no Python control flow on array values, so
the exact same reference, current, buoyancy, observation, reward and
randomization code runs in the NumPy test/eval env and in the jitted MJX env.

Frames: world is z-up (ENU-like, gravity along -z); body is FLU. Quaternions
are MuJoCo order (w, x, y, z). Free-joint qvel = [world linear velocity of the
body origin, body-frame angular velocity].
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, NamedTuple

from rov_rl.config import RandomizationConfig, RewardConfig, SimConfig
from rov_rl.vehicle import GRAVITY_MPS2, NUM_THRUSTERS, VEHICLE

Array = Any  # numpy.ndarray or jax.Array
NUM_WAYPOINTS = 4
OBS_SIZE = 29


class EpisodeParams(NamedTuple):
    """Everything randomized per episode (a pytree for JAX)."""

    is_station: Array  # 1.0 station keeping, 0.0 trajectory tracking
    waypoints_m: Array  # (NUM_WAYPOINTS, 3) world
    waypoint_yaw_rad: Array  # (NUM_WAYPOINTS,)
    ref_speed_mps: Array
    start_pos_m: Array  # (3,)
    start_quat: Array  # (4,)
    current_const_mps: Array  # (3,) world
    current_amp_mps: Array  # (3,)
    current_freq_hz: Array  # (3,)
    current_phase_rad: Array  # (3,)
    mass_scale: Array
    buoyancy_ratio: Array  # B / W of the (scaled) vehicle
    cob_offset_m: Array  # (3,) body frame, relative to CoG
    drag_scale: Array  # multiplies MuJoCo density and viscosity
    thruster_gain: Array  # (8,) healthy gain
    thruster_failed: Array  # (8,) 1.0 for the failed thruster
    failure_time_s: Array
    thruster_tau_s: Array


NUM_UNIFORMS = 50


class _Cursor:
    """Hands out consecutive slices of a uniform [0, 1) sample vector."""

    def __init__(self, u: Array) -> None:
        self._u = u
        self._i = 0

    def take(self, n: int) -> Array:
        out = self._u[self._i : self._i + n]
        self._i += n
        if self._i > NUM_UNIFORMS:
            raise ValueError("NUM_UNIFORMS too small for the sampler")
        return out

    def one(self) -> Array:
        return self.take(1)[0]


def _lerp(u: Array, lo_hi: tuple[float, float]) -> Array:
    return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * u


def yaw_quat(xp: ModuleType, roll: Array, pitch: Array, yaw: Array) -> Array:
    """Quaternion (w, x, y, z) for intrinsic Z-Y-X (yaw, pitch, roll)."""
    cr, sr = xp.cos(roll / 2), xp.sin(roll / 2)
    cp, sp = xp.cos(pitch / 2), xp.sin(pitch / 2)
    cy, sy = xp.cos(yaw / 2), xp.sin(yaw / 2)
    return xp.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def sample_episode(xp: ModuleType, u: Array, cfg: RandomizationConfig, sim: SimConfig) -> EpisodeParams:
    """Map a uniform vector u (NUM_UNIFORMS,) to episode parameters."""
    c = _Cursor(u)
    is_station = (c.one() < cfg.p_station).astype(u.dtype)

    # Tracking path: random 3D segments from the origin, random yaw per waypoint.
    az = 2.0 * math.pi * c.take(NUM_WAYPOINTS - 1)
    length = _lerp(c.take(NUM_WAYPOINTS - 1), cfg.segment_length_m)
    dz = _lerp(c.take(NUM_WAYPOINTS - 1), (-cfg.segment_dz_m, cfg.segment_dz_m))
    steps = xp.stack([length * xp.cos(az), length * xp.sin(az), dz], axis=-1)
    track_wp = xp.concatenate([xp.zeros((1, 3), dtype=u.dtype), xp.cumsum(steps, axis=0)], axis=0)
    track_yaw = _lerp(c.take(NUM_WAYPOINTS), (-math.pi, math.pi))
    waypoints = (1.0 - is_station) * track_wp
    waypoint_yaw = (1.0 - is_station) * track_yaw
    ref_speed = _lerp(c.one(), cfg.ref_speed_mps)

    start_range = is_station * cfg.station_start_pos_m + (1.0 - is_station) * cfg.track_start_pos_m
    yaw_range = is_station * cfg.station_start_yaw_rad + (1.0 - is_station) * cfg.track_start_yaw_rad
    start_pos = (2.0 * c.take(3) - 1.0) * start_range
    start_yaw = waypoint_yaw[0] + (2.0 * c.one() - 1.0) * yaw_range
    tilt = (2.0 * c.take(2) - 1.0) * cfg.start_tilt_rad
    start_quat = yaw_quat(xp, tilt[0], tilt[1], start_yaw)

    mag = _lerp(c.one(), cfg.current_const_mps)
    cur_az = 2.0 * math.pi * c.one()
    cur_el = (2.0 * c.one() - 1.0) * cfg.current_elevation_max_rad
    current_const = mag * xp.stack([xp.cos(cur_el) * xp.cos(cur_az), xp.cos(cur_el) * xp.sin(cur_az), xp.sin(cur_el)])
    # Sinusoid amplitude: full amplitude horizontally, 30 % vertically.
    amp = _lerp(c.one(), cfg.current_amp_mps) * xp.asarray([1.0, 1.0, 0.3], dtype=u.dtype)
    freq = _lerp(c.take(3), cfg.current_freq_hz)
    phase = 2.0 * math.pi * c.take(3)

    mass_scale = _lerp(c.one(), cfg.mass_scale)
    buoyancy_ratio = _lerp(c.one(), cfg.buoyancy_ratio)
    cob_z = _lerp(c.one(), cfg.cob_z_m)
    zero = xp.zeros_like(cob_z)
    cob = xp.stack([zero + VEHICLE.cob_offset_m[0], zero + VEHICLE.cob_offset_m[1], cob_z])
    drag_scale = _lerp(c.one(), cfg.drag_scale)
    gain = _lerp(c.take(NUM_THRUSTERS), cfg.thruster_gain)
    fails = (c.one() < cfg.p_thruster_failure).astype(u.dtype)
    fail_idx = xp.minimum(xp.floor(c.one() * NUM_THRUSTERS), NUM_THRUSTERS - 1)
    failed = fails * (xp.arange(NUM_THRUSTERS) == fail_idx).astype(u.dtype)
    failure_time = _lerp(c.one(), cfg.failure_time_frac) * sim.episode_duration_s
    tau = _lerp(c.one(), cfg.thruster_tau_s)

    return EpisodeParams(
        is_station=is_station,
        waypoints_m=waypoints,
        waypoint_yaw_rad=waypoint_yaw,
        ref_speed_mps=ref_speed,
        start_pos_m=start_pos,
        start_quat=start_quat,
        current_const_mps=current_const,
        current_amp_mps=amp,
        current_freq_hz=freq,
        current_phase_rad=phase,
        mass_scale=mass_scale,
        buoyancy_ratio=buoyancy_ratio,
        cob_offset_m=cob,
        drag_scale=drag_scale,
        thruster_gain=gain,
        thruster_failed=failed,
        failure_time_s=failure_time,
        thruster_tau_s=tau,
    )


def wrap_angle(xp: ModuleType, a: Array) -> Array:
    return xp.arctan2(xp.sin(a), xp.cos(a))


class Reference(NamedTuple):
    pos_m: Array  # (3,) world
    vel_mps: Array  # (3,) world
    yaw_rad: Array
    yaw_rate_radps: Array


def reference(xp: ModuleType, t_s: Array, p: EpisodeParams) -> Reference:
    """Constant-speed piecewise-linear path through the waypoints, then hold.

    Yaw is interpolated linearly (shortest way) along each segment.
    Station keeping is the degenerate case of identical waypoints.
    """
    eps = 1e-6
    seg = p.waypoints_m[1:] - p.waypoints_m[:-1]
    seg_len = xp.sqrt(xp.sum(seg**2, axis=-1))
    dur = seg_len / p.ref_speed_mps
    t_start = xp.concatenate([xp.zeros((1,), dtype=dur.dtype), xp.cumsum(dur)[:-1]])
    frac = xp.clip((t_s - t_start) / xp.maximum(dur, eps), 0.0, 1.0)
    active = ((t_s >= t_start) & (t_s < t_start + dur)).astype(dur.dtype)
    pos = p.waypoints_m[0] + xp.sum(frac[:, None] * seg, axis=0)
    vel = xp.sum((active / xp.maximum(dur, eps))[:, None] * seg, axis=0)
    dyaw = wrap_angle(xp, p.waypoint_yaw_rad[1:] - p.waypoint_yaw_rad[:-1])
    yaw = wrap_angle(xp, p.waypoint_yaw_rad[0] + xp.sum(frac * dyaw))
    yaw_rate = xp.sum(active / xp.maximum(dur, eps) * dyaw)
    return Reference(pos, vel, yaw, yaw_rate)


def path_end_time_s(xp: ModuleType, p: EpisodeParams) -> Array:
    seg = p.waypoints_m[1:] - p.waypoints_m[:-1]
    return xp.sum(xp.sqrt(xp.sum(seg**2, axis=-1))) / p.ref_speed_mps


def current_velocity(xp: ModuleType, t_s: Array, p: EpisodeParams) -> Array:
    """World-frame water velocity [m/s]; written to MuJoCo option.wind."""
    phase = 2.0 * math.pi * p.current_freq_hz * t_s + p.current_phase_rad
    return p.current_const_mps + p.current_amp_mps * xp.sin(phase)


def quat_to_mat(xp: ModuleType, q: Array) -> Array:
    """Body-to-world rotation matrix from a (w, x, y, z) quaternion (normalized here)."""
    q = q / xp.sqrt(xp.sum(q**2))
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        xp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        xp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def yaw_of(xp: ModuleType, rot: Array) -> Array:
    return xp.arctan2(rot[1, 0], rot[0, 0])


def buoyancy_n(p: EpisodeParams) -> Array:
    """Buoyancy magnitude [N] for the (mass-scaled) vehicle."""
    return p.buoyancy_ratio * p.mass_scale * VEHICLE.mass_kg * GRAVITY_MPS2


def buoyancy_wrench(xp: ModuleType, quat: Array, p: EpisodeParams) -> Array:
    """xfrc_applied row for body 'rov': world force [N] and torque [N m] about the CoG.

    Buoyancy acts straight up (world +z) at the centre of buoyancy, which sits
    at cob_offset_m in the body frame; applying it at the CoG requires the
    moment r_cob x F.
    """
    rot = quat_to_mat(xp, quat)
    b = buoyancy_n(p)
    force = xp.stack([xp.zeros_like(b), xp.zeros_like(b), b])
    torque = xp.cross(rot @ p.cob_offset_m, force)
    return xp.concatenate([force, torque])


def thruster_ctrl(xp: ModuleType, action: Array, t_s: Array, p: EpisodeParams) -> Array:
    """Normalized command in [-1, 1]^8 -> MuJoCo ctrl [N] incl. gain loss and failure."""
    healthy = 1.0 - p.thruster_failed * (t_s >= p.failure_time_s).astype(action.dtype)
    return xp.clip(action, -1.0, 1.0) * VEHICLE.thrust_max_n * p.thruster_gain * healthy


def update_integral(xp: ModuleType, integral: Array, pos_err_world: Array, sim: SimConfig) -> Array:
    decay = 1.0 - sim.ctrl_dt_s / sim.integral_tau_s
    lim = sim.integral_limit_ms
    return xp.clip(integral * decay + pos_err_world * sim.ctrl_dt_s, -lim, lim)


class Errors(NamedTuple):
    pos_err_world_m: Array  # p_ref - p
    pos_err_m: Array  # norm
    yaw_err_rad: Array  # wrapped yaw_ref - yaw
    cos_tilt: Array  # body z . world z


def tracking_errors(xp: ModuleType, qpos: Array, ref: Reference) -> Errors:
    rot = quat_to_mat(xp, qpos[3:7])
    e = ref.pos_m - qpos[:3]
    return Errors(e, xp.sqrt(xp.sum(e**2)), wrap_angle(xp, ref.yaw_rad - yaw_of(xp, rot)), rot[2, 2])


def observation(
    xp: ModuleType,
    qpos: Array,
    qvel: Array,
    ref: Reference,
    integral: Array,
    last_action: Array,
    sim: SimConfig,
) -> Array:
    """29-dim policy observation; all vectors expressed in the body frame.

    Contains no privileged information (current, mass, failures are hidden).
    """
    rot = quat_to_mat(xp, qpos[3:7])
    rt = rot.T
    err = tracking_errors(xp, qpos, ref)
    obs = xp.concatenate([
        xp.clip(rt @ err.pos_err_world_m, -2.0, 2.0),  # 3
        rt @ qvel[:3],  # 3 body linear velocity
        rt @ ref.vel_mps,  # 3 reference feed-forward velocity
        qvel[3:6],  # 3 body angular velocity
        rt @ xp.asarray([0.0, 0.0, -1.0], dtype=qpos.dtype),  # 3 gravity direction
        xp.stack([xp.sin(err.yaw_err_rad), xp.cos(err.yaw_err_rad), ref.yaw_rate_radps]),  # 3
        rt @ integral / sim.integral_limit_ms,  # 3
        last_action,  # 8
    ])
    return xp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def reward_and_done(
    xp: ModuleType,
    err: Errors,
    action: Array,
    last_action: Array,
    finite: Array,
    cfg: RewardConfig,
) -> tuple[Array, Array, dict[str, Array]]:
    """Per-step reward, termination flag and reward terms."""
    terms = {
        "pos_coarse": cfg.w_pos_coarse * xp.exp(-((err.pos_err_m / cfg.pos_coarse_scale_m) ** 2)),
        "pos_fine": cfg.w_pos_fine * xp.exp(-((err.pos_err_m / cfg.pos_fine_scale_m) ** 2)),
        "yaw": cfg.w_yaw * xp.exp(-((err.yaw_err_rad / cfg.yaw_scale_rad) ** 2)),
        "effort": -cfg.w_effort * xp.mean(xp.abs(action)),
        "action_rate": -cfg.w_action_rate * xp.mean((action - last_action) ** 2),
        "tilt": -cfg.w_tilt * (1.0 - err.cos_tilt),
    }
    failed = (err.pos_err_m > cfg.max_pos_error_m) | (err.cos_tilt < cfg.min_cos_tilt) | (~finite)
    done = failed.astype(action.dtype)
    reward = sum(terms.values()) - cfg.termination_penalty * done
    reward = xp.where(finite, reward, -cfg.termination_penalty)
    return reward, done, terms
