"""Array-backend-agnostic logic of the local-planning / obstacle-avoidance task ("nav").

Same rules as rov_rl/core.py: every function takes ``xp`` (``numpy`` or ``jax.numpy``) and has no Python
control flow on array values, so the CPU evaluation env and the jitted MJX training env run identical code.

Obstacle primitives are capsules (pillars and pipe spans), yawed boxes (rocks / wreck parts) and the seabed
plane. MJX's ray caster (mujoco-mjx 3.17 ``_src/ray.py``) supports PLANE, SPHERE, CAPSULE, ELLIPSOID, BOX and
MESH but silently skips CYLINDER, so cylinders are deliberately not used. Rays and clearances are computed
analytically here (one code path for both backends); tests check them against ``mujoco.mj_ray`` and, where
JAX is installed, ``mjx.ray`` on a compiled scene model (rov_rl/nav_cpu_env.py::build_scene_model).

The physics model is unchanged: contacts stay disabled, and a collision is the vehicle's bounding sphere
(radius VEHICLE_BOUNDING_RADIUS_M) touching an obstacle or the seabed, which ends the episode.

Frames: world z-up, body FLU, quaternions (w, x, y, z), as in core.py.
"""

from __future__ import annotations

import functools
import math
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from rov_rl import core
from rov_rl.config import (
    NavCommandConfig,
    NavRewardConfig,
    NavSceneConfig,
    RandomizationConfig,
    SimConfig,
    SonarConfig,
)

Array = Any  # numpy.ndarray or jax.Array

# Half the diagonal of the published 0.46 x 0.58 x 0.38 m envelope ([vB22] Table 5): a sphere that contains
# the whole vehicle, so a collision is never missed (slightly conservative at the corners).
VEHICLE_BOUNDING_RADIUS_M = 0.5 * math.sqrt(0.46**2 + 0.58**2 + 0.38**2)
NAV_NUM_UNIFORMS = 256
PILLAR_BURY_M = 1.0  # pillars start this far below the seabed so the seabed joint has no gap
_EPS = 1e-9
_DIR_EPS = 1e-12
_NO_HIT = 1e30  # "infinite" distance that stays finite in float32 arithmetic


class NavScene(NamedTuple):
    """One obstacle field (a pytree for JAX). Masks are 1.0 (active) / 0.0 (inactive)."""

    goal_m: Array  # (3,) world
    start_yaw_rad: Array
    seabed_z_m: Array
    density: Array  # slot occupancy probability used for this field (logging)
    cap_a_m: Array  # (C, 3) capsule segment start
    cap_b_m: Array  # (C, 3) capsule segment end
    cap_radius_m: Array  # (C,)
    cap_active: Array  # (C,)
    cap_is_pipe: Array  # (C,) 1.0 for pipe spans, 0.0 for pillars (rendering / logging only)
    box_center_m: Array  # (B, 3)
    box_half_m: Array  # (B, 3) half sizes in the box frame
    box_yaw_rad: Array  # (B,)
    box_active: Array  # (B,)


class NavEpisode(NamedTuple):
    vehicle: core.EpisodeParams  # current, plant and start pose
    scene: NavScene


class NavMeasurement(NamedTuple):
    """What the vehicle senses: the only input of both the policy and the classical baseline."""

    sonar_range_m: Array  # (num_beams,) horizontal fan, nearest echo within the vertical aperture
    altimeter_m: Array  # down beam
    up_range_m: Array  # up beam
    vel_body_mps: Array  # (3,) DVL (bottom lock), body frame
    ang_vel_radps: Array  # (3,) gyro, body frame
    gravity_body: Array  # (3,) unit gravity direction in the body frame (roll / pitch)
    yaw_rad: Array  # compass heading
    depth_m: Array  # below the nominal surface
    goal_body_m: Array  # (3,) goal relative to the vehicle, body frame (navigation solution)


class _Cursor:
    def __init__(self, u: Array) -> None:
        self._u = u
        self._i = 0

    def take(self, n: int) -> Array:
        out = self._u[self._i : self._i + n]
        self._i += n
        if self._i > self._u.shape[0]:
            raise ValueError("NAV_NUM_UNIFORMS too small for the scene sampler")
        return out

    def one(self) -> Array:
        return self.take(1)[0]


def _lerp(u: Array, lo_hi: tuple[float, float]) -> Array:
    return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * u


def _sym(u: Array, half_range: float) -> Array:
    return (2.0 * u - 1.0) * half_range


# --- small linear algebra without matmul -------------------------------------------------------------------
# No `@` / einsum anywhere in this module: on GPUs JAX's default float32 matmul precision may use TF32 (10-bit
# mantissa, ~1e-3 relative error). Ray-quadratic discriminants cancel catastrophically (b^2 - c with b ~ 10 m),
# so TF32 products turned grazing misses into hits and made MJX ranges up to ~0.9 m short of the CPU env.
# Elementwise multiply + sum is always evaluated in full precision. tests/test_nav_geometry.py guards this.


def _pair_dot(xp: ModuleType, a: Array, b: Array) -> Array:
    """(M, 3) x (C, 3) -> (M, C) pairwise dot products."""
    return xp.sum(a[:, None, :] * b[None, :, :], axis=-1)


def _mv(xp: ModuleType, m: Array, v: Array) -> Array:
    """m @ v for a (3, 3) matrix and a (3,) vector."""
    return xp.sum(m * v[None, :], axis=-1)


def _mtv(xp: ModuleType, m: Array, v: Array) -> Array:
    """m.T @ v for a (3, 3) matrix and a (3,) vector."""
    return xp.sum(m * v[:, None], axis=0)


# --- geometry: rays ----------------------------------------------------------------------------------------


def _first_positive_sphere_hit(xp: ModuleType, o: Array, d: Array, center: Array, r: Array) -> Array:
    """Entry distance of unit rays d (M, 3) from o (3,) into spheres (C,); _NO_HIT if none. Shape (M, C)."""
    oc = o[None, :] - center  # (C, 3)
    b = _pair_dot(xp, d, oc)  # (M, C)
    c = xp.sum(oc**2, axis=-1) - r**2  # (C,)
    h = b**2 - c[None, :]
    t = -b - xp.sqrt(xp.maximum(h, 0.0))
    return xp.where((h >= 0.0) & (t >= 0.0), t, _NO_HIT)


def ray_capsules(xp: ModuleType, o: Array, d: Array, a: Array, b: Array, r: Array, active: Array) -> Array:
    """Nearest entry distance of unit rays (M, 3) from o into any active capsule; (M,), _NO_HIT if none.

    A capsule is the union of a finite cylinder (segment a-b, radius r) and two end spheres; the entry into a
    union of convex parts is the smallest entry over the parts. The cylinder quadratic is formed from the
    components perpendicular to the unit axis (well conditioned in float32 for long pillars). Rays starting
    inside a capsule do not hit it (only reachable in a collision state, which ends the episode).
    """
    ba = b - a  # (C, 3)
    length = xp.sqrt(xp.sum(ba**2, axis=-1))  # (C,)
    n = ba / xp.maximum(length, _EPS)[:, None]
    oa = o[None, :] - a  # (C, 3)
    d_ax = _pair_dot(xp, d, n)  # (M, C) axial component of the ray direction
    oa_ax = xp.sum(oa * n, axis=-1)  # (C,)
    oa_perp = oa - oa_ax[:, None] * n  # (C, 3)
    qa = 1.0 - d_ax**2  # |d_perp|^2 for unit d
    qb = _pair_dot(xp, d, oa_perp)  # d_perp . oa_perp == d . oa_perp, (M, C)
    qc = xp.sum(oa_perp**2, axis=-1) - r**2  # (C,)
    h = qb**2 - qa * qc[None, :]
    parallel = qa <= _EPS
    t_body = (-qb - xp.sqrt(xp.maximum(h, 0.0))) / xp.where(parallel, 1.0, qa)
    y = oa_ax[None, :] + t_body * d_ax  # axial coordinate of the hit point
    body_ok = (~parallel) & (h >= 0.0) & (t_body >= 0.0) & (y > 0.0) & (y < length[None, :])
    t = xp.where(body_ok, t_body, _NO_HIT)
    t = xp.minimum(t, _first_positive_sphere_hit(xp, o, d, a, r))
    t = xp.minimum(t, _first_positive_sphere_hit(xp, o, d, b, r))
    t = xp.where(active[None, :] > 0.5, t, _NO_HIT)
    return xp.min(t, axis=-1)


def ray_boxes(xp: ModuleType, o: Array, d: Array, center: Array, half: Array, yaw: Array, active: Array) -> Array:
    """Nearest entry distance of unit rays (M, 3) into active yawed boxes (slab method); (M,), _NO_HIT if none."""
    c, s = xp.cos(yaw), xp.sin(yaw)  # (B,); box frame = world frame yawed about z
    rel = o[None, :] - center
    o_box = xp.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=-1)  # (B, 3)
    dx, dy = d[:, 0:1], d[:, 1:2]  # (M, 1)
    d_box = xp.stack([c[None] * dx + s[None] * dy, -s[None] * dx + c[None] * dy,
                      xp.broadcast_to(d[:, 2:3], (d.shape[0], yaw.shape[0]))], axis=-1)  # (M, B, 3)
    d_safe = xp.copysign(xp.maximum(xp.abs(d_box), _DIR_EPS), d_box)  # keeps |d| >= eps and the input dtype
    t1 = (-half[None] - o_box[None]) / d_safe
    t2 = (half[None] - o_box[None]) / d_safe
    t_near = xp.max(xp.minimum(t1, t2), axis=-1)  # (M, B)
    t_far = xp.min(xp.maximum(t1, t2), axis=-1)
    hit = (t_far >= t_near) & (t_near >= 0.0) & (active[None, :] > 0.5)
    return xp.min(xp.where(hit, t_near, _NO_HIT), axis=-1)


def ray_seabed(xp: ModuleType, o: Array, d: Array, seabed_z: Array) -> Array:
    """Distance of unit rays (M, 3) to the seabed plane z = seabed_z, seen from above; (M,)."""
    dz = d[:, 2]
    t = (seabed_z - o[2]) / xp.where(dz < -_DIR_EPS, dz, -1.0)
    return xp.where((dz < -_DIR_EPS) & (o[2] >= seabed_z), t, _NO_HIT)


def ray_scene(xp: ModuleType, o: Array, d: Array, scene: NavScene) -> Array:
    """Nearest hit distance over all obstacles and the seabed; (M,), _NO_HIT if nothing is hit."""
    t = ray_capsules(xp, o, d, scene.cap_a_m, scene.cap_b_m, scene.cap_radius_m, scene.cap_active)
    t = xp.minimum(t, ray_boxes(xp, o, d, scene.box_center_m, scene.box_half_m, scene.box_yaw_rad, scene.box_active))
    return xp.minimum(t, ray_seabed(xp, o, d, scene.seabed_z_m))


# --- geometry: clearance -----------------------------------------------------------------------------------


def capsule_distances(xp: ModuleType, p: Array, a: Array, b: Array, r: Array) -> Array:
    """Signed surface distance of point p (3,) to each capsule; (C,)."""
    ba = b - a
    t = xp.clip(xp.sum((p[None, :] - a) * ba, axis=-1) / xp.maximum(xp.sum(ba**2, axis=-1), _EPS), 0.0, 1.0)
    closest = a + t[:, None] * ba
    return xp.sqrt(xp.sum((p[None, :] - closest) ** 2, axis=-1)) - r


def box_distances(xp: ModuleType, p: Array, center: Array, half: Array, yaw: Array) -> Array:
    """Signed distance of point p (3,) to each yawed box (negative inside); (B,)."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    rel = p[None, :] - center
    local = xp.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=-1)
    q = xp.abs(local) - half
    outside = xp.sqrt(xp.sum(xp.maximum(q, 0.0) ** 2, axis=-1))
    return outside + xp.minimum(xp.max(q, axis=-1), 0.0)


def scene_distance(xp: ModuleType, p: Array, scene: NavScene) -> Array:
    """Smallest signed surface distance from point p to any active obstacle or the seabed [m]."""
    dc = capsule_distances(xp, p, scene.cap_a_m, scene.cap_b_m, scene.cap_radius_m)
    db = box_distances(xp, p, scene.box_center_m, scene.box_half_m, scene.box_yaw_rad)
    d_obs = xp.minimum(xp.min(xp.where(scene.cap_active > 0.5, dc, _NO_HIT)),
                       xp.min(xp.where(scene.box_active > 0.5, db, _NO_HIT)))
    return xp.minimum(d_obs, p[2] - scene.seabed_z_m)


def clearance_m(xp: ModuleType, pos: Array, scene: NavScene) -> Array:
    """Gap between the vehicle's bounding sphere and the nearest obstacle / seabed; < 0 means collision."""
    return scene_distance(xp, pos, scene) - VEHICLE_BOUNDING_RADIUS_M


# --- scene sampling ----------------------------------------------------------------------------------------


def num_capsules(cfg: NavSceneConfig) -> int:
    return cfg.lattice_rows * cfg.lattice_cols + cfg.num_pipes


def num_boxes(cfg: NavSceneConfig) -> int:
    return cfg.lattice_rows * cfg.lattice_cols


def sample_scene(xp: ModuleType, u: Array, cfg: NavSceneConfig) -> NavScene:
    """Map a uniform vector u (NAV_NUM_UNIFORMS,) to an obstacle field (see NavSceneConfig)."""
    c = _Cursor(u)
    dt = u.dtype
    dist = _lerp(c.one(), cfg.goal_distance_m)
    bearing = 2.0 * math.pi * c.one() - math.pi
    goal_dz = _sym(c.one(), cfg.goal_dz_m)
    start_yaw = core.wrap_angle(xp, bearing + _sym(c.one(), cfg.start_heading_offset_rad))
    seabed_z = -_lerp(c.one(), cfg.seabed_altitude_m)
    dens_u = c.one()
    slot_p = _lerp(dens_u, cfg.slot_density)
    pipe_p = _lerp(dens_u, cfg.pipe_prob)
    zero = xp.zeros_like(dist)
    along = xp.stack([xp.cos(bearing), xp.sin(bearing), zero])
    cross = xp.stack([-xp.sin(bearing), xp.cos(bearing), zero])
    goal = dist * along + xp.stack([zero, zero, goal_dz])

    # Lattice slots: pillar or box.
    k = cfg.lattice_rows * cfg.lattice_cols
    row = xp.asarray(np.arange(k) // cfg.lattice_cols, dtype=dt)
    col = xp.asarray(np.arange(k) % cfg.lattice_cols, dtype=dt)
    cell_s = (dist - 2.0 * cfg.lattice_end_margin_m) / cfg.lattice_rows
    cell_c = 2.0 * cfg.corridor_half_width_m / cfg.lattice_cols
    s = cfg.lattice_end_margin_m + (row + 0.5) * cell_s + _sym(c.take(k), cfg.cell_jitter_frac) * cell_s
    q = -cfg.corridor_half_width_m + (col + 0.5) * cell_c + _sym(c.take(k), cfg.cell_jitter_frac) * cell_c
    xy = s[:, None] * along[None, :2] + q[:, None] * cross[None, :2]  # (k, 2)
    occupied = (c.take(k) < slot_p).astype(dt)
    is_pillar = (c.take(k) < cfg.p_pillar).astype(dt)
    pillar_r = _lerp(c.take(k), cfg.pillar_radius_m)
    half_x = _lerp(c.take(k), cfg.box_half_xy_m)
    half_y = _lerp(c.take(k), cfg.box_half_xy_m)
    half_z = _lerp(c.take(k), cfg.box_half_z_m)
    box_yaw = math.pi * c.take(k)

    ones = xp.ones((k,), dtype=dt)
    pillar_a = xp.concatenate([xy, (seabed_z - PILLAR_BURY_M) * ones[:, None]], axis=-1)
    pillar_b = xp.concatenate([xy, cfg.surface_z_m * ones[:, None]], axis=-1)
    box_center = xp.concatenate([xy, (seabed_z + half_z)[:, None]], axis=-1)
    box_half = xp.stack([half_x, half_y, half_z], axis=-1)

    # Pipe spans: pipe j is placed in the along-track band [0.25 + 0.5 j / n, 0.25 + 0.5 (j + 1) / n] * dist.
    n = cfg.num_pipes
    j = xp.asarray(np.arange(n), dtype=dt)
    pipe_s = dist * (0.25 + 0.5 * (j + c.take(n)) / max(n, 1))
    pipe_q = _sym(c.take(n), cfg.pipe_cross_offset_m)
    pipe_r = _lerp(c.take(n), cfg.pipe_radius_m)
    pipe_z = xp.maximum(goal_dz * pipe_s / dist + _sym(c.take(n), cfg.pipe_dz_m), seabed_z + pipe_r)
    pipe_yaw = bearing + 0.5 * math.pi + _sym(c.take(n), cfg.pipe_yaw_jitter_rad)
    pipe_half = _lerp(c.take(n), cfg.pipe_half_length_m)
    pipe_active = (c.take(n) < pipe_p).astype(dt)
    pipe_center = (pipe_s[:, None] * along[None, :] + pipe_q[:, None] * cross[None, :]
                   + pipe_z[:, None] * xp.asarray([0.0, 0.0, 1.0], dtype=dt)[None, :])
    pipe_axis = xp.stack([xp.cos(pipe_yaw), xp.sin(pipe_yaw), xp.zeros_like(pipe_yaw)], axis=-1)

    cap_a = xp.concatenate([pillar_a, pipe_center - pipe_half[:, None] * pipe_axis], axis=0)
    cap_b = xp.concatenate([pillar_b, pipe_center + pipe_half[:, None] * pipe_axis], axis=0)
    cap_r = xp.concatenate([pillar_r, pipe_r])
    cap_active = xp.concatenate([occupied * is_pillar, pipe_active])
    cap_is_pipe = xp.concatenate([xp.zeros((k,), dtype=dt), xp.ones((n,), dtype=dt)])
    box_active = occupied * (1.0 - is_pillar)

    # Keep-out: drop obstacles whose surface is too close to the start or the goal.
    min_gap = cfg.keepout_m + VEHICLE_BOUNDING_RADIUS_M
    start = xp.zeros((3,), dtype=dt)
    cap_ok = ((capsule_distances(xp, start, cap_a, cap_b, cap_r) >= min_gap)
              & (capsule_distances(xp, goal, cap_a, cap_b, cap_r) >= min_gap)).astype(dt)
    box_ok = ((box_distances(xp, start, box_center, box_half, box_yaw) >= min_gap)
              & (box_distances(xp, goal, box_center, box_half, box_yaw) >= min_gap)).astype(dt)

    return NavScene(
        goal_m=goal,
        start_yaw_rad=start_yaw,
        seabed_z_m=seabed_z,
        density=slot_p,
        cap_a_m=cap_a,
        cap_b_m=cap_b,
        cap_radius_m=cap_r,
        cap_active=cap_active * cap_ok,
        cap_is_pipe=cap_is_pipe,
        box_center_m=box_center,
        box_half_m=box_half,
        box_yaw_rad=box_yaw,
        box_active=box_active * box_ok,
    )


def make_episode(
    xp: ModuleType,
    u_vehicle: Array,
    u_scene: Array,
    vehicle_cfg: RandomizationConfig,
    scene_cfg: NavSceneConfig,
    sim: SimConfig,
) -> NavEpisode:
    """Current and plant from core.sample_episode; start pose (origin, level, scene yaw) from the scene."""
    scene = sample_scene(xp, u_scene, scene_cfg)
    vehicle = core.sample_episode(xp, u_vehicle, vehicle_cfg, sim)
    zero = xp.zeros_like(scene.start_yaw_rad)
    vehicle = vehicle._replace(
        start_pos_m=xp.zeros((3,), dtype=u_scene.dtype),
        start_quat=core.yaw_quat(xp, zero, zero, scene.start_yaw_rad),
    )
    return NavEpisode(vehicle=vehicle, scene=scene)


# --- sensing -----------------------------------------------------------------------------------------------


def _unit_dirs(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], -1)


def beam_azimuths_rad(cfg: SonarConfig) -> np.ndarray:
    """Horizontal-fan azimuths in the body frame, right (-y) to left (+y)."""
    return np.linspace(-0.5 * cfg.fan_width_rad, 0.5 * cfg.fan_width_rad, cfg.num_beams)


@functools.lru_cache(maxsize=8)
def sonar_dirs_body(cfg: SonarConfig) -> np.ndarray:
    """Unit ray directions in the body frame: (rows * num_beams) fan rays (row-major), then down, then up."""
    az = beam_azimuths_rad(cfg)
    rows = [_unit_dirs(az, np.full_like(az, el)) for el in cfg.elevation_rows_rad]
    vertical = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    dirs = np.concatenate(rows + [vertical], axis=0)
    dirs.setflags(write=False)  # cached and shared
    return dirs


def num_noise_normals(cfg: SonarConfig) -> int:
    return cfg.num_beams + 2 + 3  # fan + vertical ranges + DVL


def num_noise_uniforms(cfg: SonarConfig) -> int:
    return cfg.num_beams + 2  # dropouts


def true_ranges(xp: ModuleType, qpos: Array, scene: NavScene, cfg: SonarConfig) -> tuple[Array, Array]:
    """Noise-free (fan (num_beams,), vertical (2,)) ranges, clipped to max_range_m."""
    rot = core.quat_to_mat(xp, qpos[3:7])
    origin = qpos[:3] + _mv(xp, rot, xp.asarray(cfg.mount_pos_m, dtype=qpos.dtype))
    dirs = _pair_dot(xp, xp.asarray(sonar_dirs_body(cfg), dtype=qpos.dtype), rot)  # R d for every body ray
    t = xp.minimum(ray_scene(xp, origin, dirs, scene), cfg.max_range_m)
    n_fan = len(cfg.elevation_rows_rad) * cfg.num_beams
    fan = xp.min(xp.reshape(t[:n_fan], (len(cfg.elevation_rows_rad), cfg.num_beams)), axis=0)
    return fan, t[n_fan:]


def measure(
    xp: ModuleType,
    qpos: Array,
    qvel: Array,
    scene: NavScene,
    cfg: SonarConfig,
    surface_z_m: float,
    noise_normal: Array,
    noise_uniform: Array,
) -> NavMeasurement:
    """Sensor readings. noise_normal (num_noise_normals,) ~ N(0, 1), noise_uniform (num_noise_uniforms,) ~ U[0, 1)."""
    rot = core.quat_to_mat(xp, qpos[3:7])
    fan, vertical = true_ranges(xp, qpos, scene, cfg)
    ranges = xp.concatenate([fan, vertical])
    nb = cfg.num_beams
    sigma = cfg.range_noise_std_m + cfg.range_noise_frac * ranges
    noisy = ranges + sigma * noise_normal[: nb + 2]
    noisy = xp.where(noise_uniform < cfg.dropout_prob, cfg.max_range_m, noisy)
    noisy = xp.clip(noisy, 0.0, cfg.max_range_m)
    vel_body = _mtv(xp, rot, qvel[:3]) + cfg.dvl_noise_std_mps * noise_normal[nb + 2 : nb + 5]
    return NavMeasurement(
        sonar_range_m=noisy[:nb],
        altimeter_m=noisy[nb],
        up_range_m=noisy[nb + 1],
        vel_body_mps=vel_body,
        ang_vel_radps=qvel[3:6],
        gravity_body=_mtv(xp, rot, xp.asarray([0.0, 0.0, -1.0], dtype=qpos.dtype)),
        yaw_rad=core.yaw_of(xp, rot),
        depth_m=surface_z_m - qpos[2],
        goal_body_m=_mtv(xp, rot, scene.goal_m - qpos[:3]),
    )


GOAL_DISTANCE_SCALE_M = 30.0  # observation scaling only


def nav_obs_size(cfg: SonarConfig) -> int:
    return cfg.num_beams + 22


def nav_observation(xp: ModuleType, meas: NavMeasurement, last_cmd: Array, cfg: SonarConfig,
                    surface_z_m: float) -> Array:
    """Policy observation (nav_obs_size,): a fixed encoding of NavMeasurement plus the previous command."""
    g = meas.goal_body_m
    g_norm = xp.sqrt(xp.sum(g**2))
    obs = xp.concatenate([
        meas.sonar_range_m / cfg.max_range_m,  # num_beams
        xp.stack([meas.altimeter_m, meas.up_range_m]) / cfg.max_range_m,  # 2
        meas.vel_body_mps,  # 3
        meas.ang_vel_radps,  # 3
        meas.gravity_body,  # 3
        xp.stack([xp.sin(meas.yaw_rad), xp.cos(meas.yaw_rad), meas.depth_m / surface_z_m]),  # 3
        g / xp.maximum(g_norm, 1.0),  # 3 goal direction, shrinking inside 1 m
        xp.stack([xp.minimum(g_norm, GOAL_DISTANCE_SCALE_M) / GOAL_DISTANCE_SCALE_M]),  # 1
        last_cmd,  # 4
    ])
    return xp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# --- low-level velocity control (shared by RL and the classical baseline) ----------------------------------


class VelocityControllerParams(NamedTuple):
    """Constants of the velocity-level controller; numbers come from PidGains and VehicleParams."""

    tam_pinv: Array  # (8, 6)
    mass_kg: float
    inertia_kgm2: Array  # (3,)
    lin_damp: Array  # (3,)
    quad_damp: Array  # (3,)
    net_buoyancy_n: float
    thrust_max_n: float
    kp_vel_ps: float
    kp_yaw_rate_ps: float
    kp_att_ps2: float
    kd_att_ps: float


def command_scale(cfg: NavCommandConfig) -> np.ndarray:
    return np.array([cfg.surge_max_mps, cfg.sway_max_mps, cfg.heave_max_mps, cfg.yaw_rate_max_radps])


def velocity_wrench(xp: ModuleType, qpos: Array, qvel: Array, v_cmd_world: Array, r_cmd: Array,
                    vc: VelocityControllerParams) -> Array:
    """Body wrench (6,) tracking a world velocity and a yaw rate.

    Identical to the inner loops of rov_rl.pid_baseline.CascadePid.wrench (velocity P + nominal drag and
    buoyancy feed-forward, yaw-rate P, roll / pitch PD); tests/test_nav_baseline.py checks the equivalence.
    """
    rot = core.quat_to_mat(xp, qpos[3:7])
    v_cmd_body = _mtv(xp, rot, v_cmd_world)
    v_err_body = v_cmd_body - _mtv(xp, rot, qvel[:3])
    drag_ff = vc.lin_damp * v_cmd_body + vc.quad_damp * xp.abs(v_cmd_body) * v_cmd_body
    force = vc.mass_kg * vc.kp_vel_ps * v_err_body + drag_ff
    force = force + _mtv(xp, rot, xp.asarray([0.0, 0.0, -vc.net_buoyancy_n], dtype=qpos.dtype))
    roll = xp.arctan2(rot[2, 1], rot[2, 2])
    pitch = -xp.arcsin(xp.clip(rot[2, 0], -1.0, 1.0))
    torque = xp.stack([
        -vc.inertia_kgm2[0] * (vc.kp_att_ps2 * roll + vc.kd_att_ps * qvel[3]),
        -vc.inertia_kgm2[1] * (vc.kp_att_ps2 * pitch + vc.kd_att_ps * qvel[4]),
        vc.inertia_kgm2[2] * vc.kp_yaw_rate_ps * (r_cmd - qvel[5]),
    ])
    return xp.concatenate([force, torque])


def command_to_thrust(xp: ModuleType, qpos: Array, qvel: Array, cmd: Array, cmd_cfg: NavCommandConfig,
                      vc: VelocityControllerParams) -> Array:
    """Normalized velocity command (4,) -> normalized thruster command in [-1, 1]^8.

    cmd = (surge, sway, heave, yaw rate) / limits. Surge and sway are horizontal, along / across the current
    heading; heave is world vertical. Saturated allocations are scaled down uniformly (direction preserved).
    """
    phys = xp.clip(cmd, -1.0, 1.0) * xp.asarray(command_scale(cmd_cfg), dtype=qpos.dtype)
    yaw = core.yaw_of(xp, core.quat_to_mat(xp, qpos[3:7]))
    c, s = xp.cos(yaw), xp.sin(yaw)
    v_world = xp.stack([c * phys[0] - s * phys[1], s * phys[0] + c * phys[1], phys[2]])
    thrust = xp.sum(vc.tam_pinv * velocity_wrench(xp, qpos, qvel, v_world, phys[3], vc)[None, :], axis=-1)
    peak = xp.max(xp.abs(thrust))
    scale = xp.where(peak > vc.thrust_max_n, vc.thrust_max_n / xp.maximum(peak, _EPS), 1.0)
    return thrust * scale / vc.thrust_max_n


# --- reward ------------------------------------------------------------------------------------------------


class NavOutcome(NamedTuple):
    done: Array  # 1.0 when the episode ends (success, collision, tilt or non-finite state)
    success: Array  # reached the goal without a collision on the same step
    collision: Array
    failure: Array  # tilt beyond the envelope or non-finite state


def nav_reward_and_done(
    xp: ModuleType,
    dist_prev_m: Array,
    dist_now_m: Array,
    clearance: Array,
    cos_tilt: Array,
    thrust_action: Array,
    cmd: Array,
    last_cmd: Array,
    finite: Array,
    cfg: NavRewardConfig,
) -> tuple[Array, NavOutcome, dict[str, Array]]:
    """Per-step reward, outcome flags and reward terms. Timeouts are handled by the caller."""
    dt = thrust_action.dtype
    near = 1.0 - xp.clip(clearance, 0.0, cfg.safe_clearance_m) / cfg.safe_clearance_m
    terms = {
        "progress": cfg.w_progress_per_m * xp.nan_to_num(dist_prev_m - dist_now_m),
        "clearance": -cfg.w_clearance * near**2,
        "effort": -cfg.w_effort * xp.mean(xp.abs(thrust_action)),
        "action_rate": -cfg.w_action_rate * xp.mean((cmd - last_cmd) ** 2),
        "time": -cfg.w_time * xp.ones((), dtype=dt),
        "tilt": -cfg.w_tilt * (1.0 - xp.nan_to_num(cos_tilt, nan=1.0)),
    }
    collision = (clearance < 0.0) & finite
    failure = (cos_tilt < cfg.min_cos_tilt) | (~finite)
    failed = collision | failure
    success = (dist_now_m < cfg.goal_radius_m) & (~failed)
    reward = (sum(terms.values()) + cfg.success_bonus * success.astype(dt)
              - cfg.collision_penalty * failed.astype(dt))
    reward = xp.where(finite, reward, -cfg.collision_penalty)
    outcome = NavOutcome(
        done=(success | failed).astype(dt),
        success=success.astype(dt),
        collision=collision.astype(dt),
        failure=failure.astype(dt),
    )
    return reward, outcome, terms
