"""Classical local planner for the nav task: VFH+ over a sonar-built occupancy grid.

Input is exactly the NavMeasurement the RL policy's observation is built from (noisy sonar fan, altimeter,
up beam, DVL, attitude, compass, depth, goal vector); the planner never reads simulator state. Output is the
same normalized velocity command (surge, sway, heave, yaw rate) the policy produces, executed by the same
velocity controller (rov_rl/nav_core.py::command_to_thrust).

Pipeline per control step:
  1. Pose in a goal-centred frame with world axes: p = -R(roll, pitch, yaw) @ goal_body. This is the same
     navigation solution the policy receives as the body-frame goal vector.
  2. Log-odds occupancy grid (2D, horizontal): every fan beam clears the cells it passes and marks its echo
     cell occupied. An imaging sonar cannot resolve elevation, so echoes are placed in the horizontal plane.
     Echoes at or beyond the range where the lowest elevation row meets the seabed (from the altimeter) are
     treated as seabed, not obstacles.
  3. VFH+ (Ulrich & Borenstein, ICRA 1998): polar histogram of occupied cells inside a window, enlarged by
     vehicle radius + safety margin; binary histogram with hysteresis; candidate directions at valley
     borders / centres / the goal direction; cost = goal, current-heading and previous-choice deviation.
  4. Turn towards the chosen direction (yaw-rate P), surge when roughly aligned, slowed by the nearest
     forward sonar return; back off when an echo is inside the stop range. Depth: P on the goal height,
     with altimeter floor and overhead (up beam) ceiling.

The grid gives the baseline memory of obstacles that left the 120 deg fan; the MLP policy has none (see
README). Parameters are in config/nav_baseline.json, tuned by scripts/tune_nav_baseline.py on episode seeds
disjoint from eval_nav.py.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import numpy as np

from rov_rl import nav_core
from rov_rl.config import NavCommandConfig, SimConfig, SonarConfig

DEFAULT_VFH_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "nav_baseline.json"


@dataclasses.dataclass(frozen=True)
class VfhParams:
    # Occupancy grid.
    cell_m: float = 0.25
    grid_half_extent_m: float = 45.0  # covers start-goal distances up to 30 m plus detours
    logodds_hit: float = 0.85
    logodds_miss: float = 0.4
    logodds_min: float = -2.0
    logodds_max: float = 3.5
    occupied_logodds: float = 0.5
    free_margin_m: float = 0.3  # free-space clearing stops this far short of an echo
    seabed_gate_frac: float = 0.9
    # Optional forgetting for moving obstacles: every control step all log-odds relax towards 0 (unknown) by the
    # factor exp(-logodds_decay_per_s * dt). 0 = off (the static-scene baseline). It also forgets static obstacles
    # that left the fan, so it is tuned separately (scripts/tune_nav_baseline.py --decay-only).
    logodds_decay_per_s: float = 0.0
    # Polar histogram.
    num_sectors: int = 72
    window_radius_m: float = 6.0
    safety_margin_m: float = 0.6  # added to the vehicle bounding radius for the enlargement angle
    hist_low: float = 1.0
    hist_high: float = 2.0
    wide_valley_sectors: int = 12
    mu_goal: float = 5.0
    mu_heading: float = 2.0
    mu_prev: float = 2.0
    # Motion.
    # Speed gain before the obstacle / heading slow-down factors [m/s]. The command saturates at
    # NavCommandConfig.surge_max_mps, so a value above that limit only makes the slow-down ramp reach full
    # speed earlier.
    cruise_speed_mps: float = 0.5
    forward_cone_rad: float = 0.35  # +- half-angle of the fan used for the speed limit
    slow_range_m: float = 4.0
    stop_range_m: float = 0.5  # beyond the vehicle bounding radius
    backoff_speed_mps: float = 0.15
    k_yaw_ps: float = 1.2
    turn_in_place_rad: float = 0.8
    goal_speed_gain_ps: float = 0.3
    goal_min_speed_mps: float = 0.1
    k_depth_ps: float = 0.5
    min_altitude_m: float = 1.2
    min_overhead_m: float = 1.0

    @staticmethod
    def load(path: pathlib.Path = DEFAULT_VFH_PATH) -> "VfhParams":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(VfhParams)}
        unknown = set(raw["params"]) - known
        if unknown:
            raise ValueError(f"unknown VFH parameter keys in {path}: {sorted(unknown)}")
        params = VfhParams(**raw["params"])
        params.validate()
        return params

    def save(self, path: pathlib.Path, note: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"note": note, "params": dataclasses.asdict(self)}, f, indent=2)
            f.write("\n")

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value):
                raise ValueError(f"VFH parameter {f.name}={value} must be finite")
        positive = ("cell_m", "grid_half_extent_m", "window_radius_m", "num_sectors", "wide_valley_sectors",
                    "cruise_speed_mps", "slow_range_m", "k_yaw_ps", "k_depth_ps")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"VFH parameter {name} must be > 0")
        if not self.hist_low <= self.hist_high:
            raise ValueError("hist_low must be <= hist_high")
        if not self.logodds_min < self.occupied_logodds < self.logodds_max:
            raise ValueError("occupied_logodds must lie inside (logodds_min, logodds_max)")
        if self.logodds_decay_per_s < 0.0:
            raise ValueError("logodds_decay_per_s must be >= 0")
        if self.slow_range_m <= self.stop_range_m:
            raise ValueError("slow_range_m must exceed stop_range_m")


def rotation_from_measurement(meas: nav_core.NavMeasurement) -> np.ndarray:
    """Body-to-world rotation from the gravity direction (roll, pitch) and the compass yaw."""
    g = np.asarray(meas.gravity_body, dtype=float)  # = R^T (0, 0, -1) = -(third row of R)
    roll = math.atan2(-g[1], -g[2])
    pitch = math.asin(float(np.clip(g[0], -1.0, 1.0)))
    yaw = float(meas.yaw_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


class OccupancyGrid:
    """2D log-odds grid in the goal-centred navigation frame (world axes, origin at the goal)."""

    def __init__(self, params: VfhParams) -> None:
        self.p = params
        self.n = int(math.ceil(2.0 * params.grid_half_extent_m / params.cell_m))
        self.logodds = np.zeros((self.n, self.n))

    def reset(self) -> None:
        self.logodds[:] = 0.0

    def cell_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(flat index, in-bounds mask) of points (..., 2)."""
        ij = np.floor((xy + self.p.grid_half_extent_m) / self.p.cell_m).astype(np.int64)
        ok = np.all((ij >= 0) & (ij < self.n), axis=-1)
        ij = np.clip(ij, 0, self.n - 1)
        return ij[..., 0] * self.n + ij[..., 1], ok

    def cell_centers(self, flat: np.ndarray) -> np.ndarray:
        i, j = np.divmod(flat, self.n)
        return (np.stack([i, j], axis=-1) + 0.5) * self.p.cell_m - self.p.grid_half_extent_m

    def update(self, origin_xy: np.ndarray, dirs_xy: np.ndarray, ranges_m: np.ndarray, is_echo: np.ndarray,
               max_range_m: float, dt_s: float = 0.0) -> None:
        """Decay all cells over dt_s (if enabled), clear cells along each beam (up to the echo minus a margin, or to
        max range) and mark echo cells."""
        p = self.p
        if p.logodds_decay_per_s > 0.0 and dt_s > 0.0:
            self.logodds *= math.exp(-p.logodds_decay_per_s * dt_s)
        steps = np.arange(0.0, max_range_m, 0.5 * p.cell_m)
        free_len = np.where(is_echo, ranges_m - p.free_margin_m, ranges_m)
        pts = origin_xy[None, None, :] + steps[None, :, None] * dirs_xy[:, None, :]
        free_flat, free_ok = self.cell_index(pts[steps[None, :] < free_len[:, None]])
        hit_flat, hit_ok = self.cell_index(origin_xy[None, :] + ranges_m[:, None] * dirs_xy)
        hits = np.unique(hit_flat[hit_ok & is_echo])
        free = np.setdiff1d(np.unique(free_flat[free_ok]), hits, assume_unique=True)
        g = self.logodds.reshape(-1)
        g[free] = np.maximum(g[free] - p.logodds_miss, p.logodds_min)
        g[hits] = np.minimum(g[hits] + p.logodds_hit, p.logodds_max)

    def occupied_near(self, center_xy: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Centres (K, 2) and log-odds (K,) of occupied cells within radius_m of center_xy."""
        p = self.p
        lo, _ = self.cell_index(center_xy - radius_m)
        hi, _ = self.cell_index(center_xy + radius_m)
        i0, j0 = divmod(int(lo), self.n)
        i1, j1 = divmod(int(hi), self.n)
        sub = self.logodds[i0 : i1 + 1, j0 : j1 + 1]
        ii, jj = np.nonzero(sub > p.occupied_logodds)
        flat = (ii + i0) * self.n + (jj + j0)
        centers = self.cell_centers(flat)
        near = np.sum((centers - center_xy) ** 2, axis=-1) <= radius_m**2
        return centers[near], sub[ii, jj][near]


class VfhPlanner:
    """VFH+ local planner; act(meas) -> normalized velocity command (4,)."""

    def __init__(self, params: VfhParams, sonar: SonarConfig, command: NavCommandConfig,
                 ctrl_dt_s: float = SimConfig().ctrl_dt_s) -> None:
        params.validate()
        self.p = params
        self.ctrl_dt_s = ctrl_dt_s
        self.sonar = sonar
        self.command = command
        self.grid = OccupancyGrid(params)
        dirs = nav_core.sonar_dirs_body(sonar)
        mid_row = int(np.argmin(np.abs(np.asarray(sonar.elevation_rows_rad))))
        self._fan_dirs_body = dirs[mid_row * sonar.num_beams : (mid_row + 1) * sonar.num_beams]
        lowest = min(sonar.elevation_rows_rad)
        self._sin_lowest = math.sin(-lowest) if lowest < 0.0 else 0.0
        self._forward = np.abs(nav_core.beam_azimuths_rad(sonar)) <= params.forward_cone_rad
        self._sector_angles = -math.pi + (np.arange(params.num_sectors) + 0.5) * (2.0 * math.pi / params.num_sectors)
        self._binary = np.zeros(params.num_sectors, dtype=bool)
        self._prev_dir: float | None = None

    def reset(self) -> None:
        self.grid.reset()
        self._binary[:] = False
        self._prev_dir = None

    # --- perception ----------------------------------------------------------------------------------------
    def echo_mask(self, meas: nav_core.NavMeasurement) -> np.ndarray:
        """Fan returns treated as obstacles: an echo below max range that is not explained by the seabed."""
        r = np.asarray(meas.sonar_range_m, dtype=float)
        echo = r < self.sonar.max_range_m - 1e-6
        if self._sin_lowest > 0.0:
            seabed_range = float(meas.altimeter_m) / self._sin_lowest
            echo &= r < self.p.seabed_gate_frac * seabed_range
        return echo

    def integrate(self, meas: nav_core.NavMeasurement) -> tuple[np.ndarray, np.ndarray]:
        """Update the grid; returns (rotation, vehicle position in the goal-centred frame)."""
        rot = rotation_from_measurement(meas)
        pos = -(rot @ np.asarray(meas.goal_body_m, dtype=float))
        origin = pos + rot @ np.asarray(self.sonar.mount_pos_m)
        d = self._fan_dirs_body @ rot.T
        dxy = d[:, :2] / np.maximum(np.linalg.norm(d[:, :2], axis=-1, keepdims=True), 1e-9)
        self.grid.update(origin[:2], dxy, np.asarray(meas.sonar_range_m, dtype=float), self.echo_mask(meas),
                         self.sonar.max_range_m, self.ctrl_dt_s)
        return rot, pos

    # --- VFH+ ----------------------------------------------------------------------------------------------
    def polar_histogram(self, pos_xy: np.ndarray) -> np.ndarray:
        p = self.p
        centers, logodds = self.grid.occupied_near(pos_xy, p.window_radius_m)
        hist = np.zeros(p.num_sectors)
        if centers.shape[0] == 0:
            return hist
        rel = centers - pos_xy
        dist = np.maximum(np.linalg.norm(rel, axis=-1), 1e-6)
        beta = np.arctan2(rel[:, 1], rel[:, 0])
        certainty = np.clip((logodds - p.occupied_logodds) / (p.logodds_max - p.occupied_logodds), 0.0, 1.0)
        magnitude = certainty**2 * (2.0 - (dist / p.window_radius_m) ** 2)  # 2 at the vehicle, 1 at the rim
        enlarge = np.arcsin(np.clip((nav_core.VEHICLE_BOUNDING_RADIUS_M + p.safety_margin_m) / dist, 0.0, 1.0))
        inside = np.abs(_wrap(self._sector_angles[:, None] - beta[None, :])) <= enlarge[None, :]
        return inside.astype(float) @ magnitude

    def binary_histogram(self, hist: np.ndarray) -> np.ndarray:
        self._binary = np.where(hist > self.p.hist_high, True, np.where(hist < self.p.hist_low, False, self._binary))
        return self._binary

    def choose_direction(self, blocked: np.ndarray, goal_dir: float, heading: float) -> float | None:
        """VFH+ candidate selection; None if every sector is blocked."""
        p = self.p
        k = p.num_sectors
        width = 2.0 * math.pi / k
        if not np.any(blocked):
            candidates = [goal_dir]
        elif np.all(blocked):
            return None
        else:
            start = int(np.flatnonzero(blocked)[0])  # rotate so that the sequence starts blocked
            order = (start + np.arange(k)) % k
            free = ~blocked[order]
            candidates = []
            goal_sector = int(math.floor((goal_dir + math.pi) / width)) % k
            i = 0
            while i < k:
                if not free[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < k and free[j + 1]:
                    j += 1
                lo, hi = order[i], order[j]  # valley spans sectors lo .. hi (counter-clockwise)
                length = j - i + 1
                if length > p.wide_valley_sectors:
                    half = p.wide_valley_sectors / 2.0
                    candidates.append(float(_wrap(self._sector_angles[lo] + half * width)))
                    candidates.append(float(_wrap(self._sector_angles[hi] - half * width)))
                    offset = (goal_sector - lo) % k
                    if half <= offset <= length - 1 - half:
                        candidates.append(goal_dir)
                else:
                    candidates.append(float(_wrap(self._sector_angles[lo] + 0.5 * (length - 1) * width)))
                i = j + 1
        prev = self._prev_dir if self._prev_dir is not None else heading
        costs = [p.mu_goal * abs(_wrap(c - goal_dir)) + p.mu_heading * abs(_wrap(c - heading))
                 + p.mu_prev * abs(_wrap(c - prev)) for c in candidates]
        return candidates[int(np.argmin(costs))]

    # --- command -------------------------------------------------------------------------------------------
    def act(self, meas: nav_core.NavMeasurement) -> np.ndarray:
        p = self.p
        rot, pos = self.integrate(meas)
        heading = float(meas.yaw_rad)
        goal_rel = -pos  # goal minus vehicle, world axes
        goal_dir = math.atan2(goal_rel[1], goal_rel[0])
        dist_xy = float(np.linalg.norm(goal_rel[:2]))
        blocked = self.binary_histogram(self.polar_histogram(pos[:2]))
        direction = self.choose_direction(blocked, goal_dir, heading)

        ranges = np.asarray(meas.sonar_range_m, dtype=float)
        echo = self.echo_mask(meas)
        front = float(np.min(np.where(echo & self._forward, ranges, self.sonar.max_range_m)))
        clearance_front = front - nav_core.VEHICLE_BOUNDING_RADIUS_M
        if direction is None:
            yaw_err = float(_wrap(goal_dir - heading))
            surge = 0.0
            yaw_rate = math.copysign(0.5 * self.command.yaw_rate_max_radps, yaw_err)
        else:
            self._prev_dir = direction
            yaw_err = float(_wrap(direction - heading))
            yaw_rate = p.k_yaw_ps * yaw_err
            if abs(yaw_err) < p.turn_in_place_rad:
                slow = np.clip((clearance_front - p.stop_range_m) / (p.slow_range_m - p.stop_range_m), 0.0, 1.0)
                surge = p.cruise_speed_mps * float(slow) * math.cos(yaw_err)
                surge = min(surge, p.goal_speed_gain_ps * dist_xy + p.goal_min_speed_mps)
            else:
                surge = 0.0
        if clearance_front < p.stop_range_m:
            surge = -p.backoff_speed_mps

        heave = p.k_depth_ps * float(goal_rel[2])
        altitude = float(meas.altimeter_m)
        if altitude < p.min_altitude_m:
            heave = max(heave, p.k_depth_ps * (p.min_altitude_m - altitude))
        overhead = float(meas.up_range_m)
        if overhead < p.min_overhead_m:
            heave = min(heave, -p.k_depth_ps * (p.min_overhead_m - overhead))

        cmd = np.array([surge, 0.0, heave, yaw_rate]) / nav_core.command_scale(self.command)
        cmd = np.clip(cmd, -1.0, 1.0)
        if not np.all(np.isfinite(cmd)):
            raise FloatingPointError(f"VFH planner produced a non-finite command {cmd}")
        return cmd

