"""Configuration dataclasses: timing, reward, randomization presets, metrics."""

from __future__ import annotations

import dataclasses

from rov_rl.vehicle import DEFAULT_BUOYANCY_RATIO, VEHICLE


@dataclasses.dataclass(frozen=True)
class SimConfig:
    sim_dt_s: float = 0.004  # must equal MJCF option/timestep (checked at env init)
    ctrl_dt_s: float = 0.04  # 25 Hz control
    episode_duration_s: float = 40.0
    # Leaky integral of world-frame position error fed to the policy, so a
    # memoryless policy can reject constant currents (PI-like state).
    integral_tau_s: float = 10.0
    integral_limit_ms: float = 2.0  # [m s]

    @property
    def n_substeps(self) -> int:
        n = round(self.ctrl_dt_s / self.sim_dt_s)
        if abs(n * self.sim_dt_s - self.ctrl_dt_s) > 1e-9 or n < 1:
            raise ValueError("ctrl_dt_s must be an integer multiple of sim_dt_s")
        return n

    @property
    def episode_steps(self) -> int:
        return round(self.episode_duration_s / self.ctrl_dt_s)


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    pos_coarse_scale_m: float = 0.5
    pos_fine_scale_m: float = 0.1
    yaw_scale_rad: float = 0.3
    w_pos_coarse: float = 1.0
    w_pos_fine: float = 0.5
    w_yaw: float = 0.3
    w_effort: float = 0.05  # mean |normalized thrust|
    w_action_rate: float = 0.05  # mean squared change of normalized command
    w_tilt: float = 0.5  # (1 - cos(tilt))
    termination_penalty: float = 10.0
    # Termination conditions.
    max_pos_error_m: float = 4.0
    min_cos_tilt: float = 0.5  # 60 deg roll/pitch envelope


@dataclasses.dataclass(frozen=True)
class RandomizationConfig:
    """Per-episode sampling ranges. (lo, hi) tuples are uniform ranges."""

    p_station: float = 0.5
    # Station keeping: start offset from the hold point.
    station_start_pos_m: float = 1.0
    station_start_yaw_rad: float = 0.8
    # Trajectory tracking: start offset from the first waypoint, path shape.
    track_start_pos_m: float = 0.3
    track_start_yaw_rad: float = 0.3
    start_tilt_rad: float = 0.1
    segment_length_m: tuple[float, float] = (1.5, 3.0)
    segment_dz_m: float = 0.5
    ref_speed_mps: tuple[float, float] = (0.2, 0.4)
    # Water current [m/s], world frame: constant + per-axis sinusoid.
    current_const_mps: tuple[float, float] = (0.0, 0.35)
    current_elevation_max_rad: float = 0.15
    current_amp_mps: tuple[float, float] = (0.0, 0.12)
    current_freq_hz: tuple[float, float] = (0.02, 0.2)
    # Plant variation.
    mass_scale: tuple[float, float] = (1.0, 1.0)
    buoyancy_ratio: tuple[float, float] = (DEFAULT_BUOYANCY_RATIO, DEFAULT_BUOYANCY_RATIO)
    cob_z_m: tuple[float, float] = (VEHICLE.cob_offset_m[2], VEHICLE.cob_offset_m[2])
    drag_scale: tuple[float, float] = (1.0, 1.0)
    thruster_gain: tuple[float, float] = (1.0, 1.0)
    thruster_tau_s: tuple[float, float] = (VEHICLE.thruster_tau_s, VEHICLE.thruster_tau_s)
    p_thruster_failure: float = 0.0
    failure_time_frac: tuple[float, float] = (0.0, 0.5)  # of episode duration


TRAIN_RANDOMIZATION = RandomizationConfig(
    mass_scale=(0.85, 1.15),
    buoyancy_ratio=(0.98, 1.04),
    cob_z_m=(0.005, 0.02),
    drag_scale=(0.7, 1.4),
    thruster_gain=(0.85, 1.0),
    thruster_tau_s=(0.02, 0.08),
    p_thruster_failure=0.15,
)

# Disturbance levels for the evaluation table: (constant, sinusoid amplitude) [m/s].
# Level 4 exceeds the training range of the constant current (out of distribution).
DISTURBANCE_LEVELS_MPS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.1, 0.03),
    (0.2, 0.06),
    (0.3, 0.09),
    (0.4, 0.12),
)


def eval_randomization(task: str, level: int, scenario: str = "nominal") -> RandomizationConfig:
    """Evaluation sampling config for one task, disturbance level and robustness scenario.

    task: "station" | "track". scenario: "nominal" | "params" | "thruster_failure" | "combined".
    """
    if task not in ("station", "track"):
        raise ValueError(f"unknown task '{task}'")
    if not 0 <= level < len(DISTURBANCE_LEVELS_MPS):
        raise ValueError(f"level must be in [0, {len(DISTURBANCE_LEVELS_MPS) - 1}]")
    const, amp = DISTURBANCE_LEVELS_MPS[level]
    cfg = RandomizationConfig(
        p_station=1.0 if task == "station" else 0.0,
        current_const_mps=(const, const),
        current_amp_mps=(amp, amp),
    )
    if scenario == "nominal":
        return cfg
    params = dict(
        mass_scale=TRAIN_RANDOMIZATION.mass_scale,
        buoyancy_ratio=TRAIN_RANDOMIZATION.buoyancy_ratio,
        cob_z_m=TRAIN_RANDOMIZATION.cob_z_m,
        drag_scale=TRAIN_RANDOMIZATION.drag_scale,
        thruster_gain=TRAIN_RANDOMIZATION.thruster_gain,
        thruster_tau_s=TRAIN_RANDOMIZATION.thruster_tau_s,
    )
    failure = dict(p_thruster_failure=1.0, failure_time_frac=(0.25, 0.25))
    if scenario == "params":
        return dataclasses.replace(cfg, **params)
    if scenario == "thruster_failure":
        return dataclasses.replace(cfg, **failure)
    if scenario == "combined":
        return dataclasses.replace(cfg, **params, **failure)
    raise ValueError(f"unknown scenario '{scenario}'")


@dataclasses.dataclass(frozen=True)
class MetricConfig:
    settle_pos_m: float = 0.1
    settle_yaw_rad: float = 0.0873  # 5 deg
    # Station keeping: error statistics over t >= this (excludes initial transient).
    station_window_start_s: float = 10.0
    # Success thresholds.
    station_success_pos_m: float = 0.2  # max error in the window
    track_success_rms_m: float = 0.3
    track_success_final_m: float = 0.3


# --------------------------------------------------------------------------------------------------------------
# Local planning / obstacle avoidance task ("nav"). See rov_rl/nav_core.py and the README section "Navigation".
# --------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SonarConfig:
    """Forward-looking imaging-sonar approximation plus vertical beams, all from one body-frame mount point.

    The horizontal fan has `num_beams` azimuths. Each azimuth is cast at every elevation in
    `elevation_rows_rad` and reports the nearest return: an imaging sonar cannot resolve elevation inside its
    vertical aperture, so a beam reports the closest echo anywhere in that aperture. Beam index 0 is the
    rightmost beam (body -y), the last index the leftmost. Two vertical beams: down (altimeter) and up.
    """

    num_beams: int = 48
    fan_width_rad: float = 2.0943951  # 120 deg
    elevation_rows_rad: tuple[float, ...] = (-0.17453293, -0.08726646, 0.0, 0.08726646, 0.17453293)  # +-10 deg
    max_range_m: float = 12.0
    mount_pos_m: tuple[float, float, float] = (0.20, 0.0, 0.0)  # body FLU, front of the frame
    # Range noise: std = range_noise_std_m + range_noise_frac * range. A dropout reports "no echo" (max range).
    range_noise_std_m: float = 0.05
    range_noise_frac: float = 0.01
    dropout_prob: float = 0.03
    dvl_noise_std_mps: float = 0.02


@dataclasses.dataclass(frozen=True)
class NavSceneConfig:
    """Per-episode obstacle-field sampling ranges. (lo, hi) tuples are uniform ranges.

    Start is the world origin. The goal lies `goal_distance_m` away at a random bearing. Obstacles sit on a
    jittered lattice (`lattice_rows` x `lattice_cols`) inside a corridor of half-width `corridor_half_width_m`
    around the start-goal line; each lattice slot is occupied with probability `slot_density` and holds a
    pillar (vertical capsule from the seabed to the surface) or a rock / wreck box resting on the seabed. Up to
    two pipe spans (horizontal capsules) cross the corridor. The corridor is open at the sides, so a path
    around the field always exists. Obstacles closer than `keepout_m` (surface distance, beyond the vehicle's
    bounding radius) to the start or goal are removed.
    """

    goal_distance_m: tuple[float, float] = (15.0, 30.0)
    start_heading_offset_rad: float = 0.6  # start yaw = goal bearing +- this
    goal_dz_m: float = 1.0
    seabed_altitude_m: tuple[float, float] = (2.5, 4.5)  # seabed depth below the start
    surface_z_m: float = 30.0  # nominal water surface above the start (= start depth 30 m); pillars reach it
    corridor_half_width_m: float = 5.0
    lattice_rows: int = 6
    lattice_cols: int = 4
    lattice_end_margin_m: float = 3.0  # along-track distance of the lattice from start and goal
    cell_jitter_frac: float = 0.35  # +- fraction of the cell size
    p_pillar: float = 0.5
    pillar_radius_m: tuple[float, float] = (0.2, 0.6)
    box_half_xy_m: tuple[float, float] = (0.3, 1.2)
    box_half_z_m: tuple[float, float] = (0.3, 2.5)
    num_pipes: int = 2
    pipe_radius_m: tuple[float, float] = (0.2, 0.4)
    pipe_half_length_m: tuple[float, float] = (4.0, 8.0)
    pipe_dz_m: float = 1.0  # pipe axis height relative to the start-goal line
    pipe_cross_offset_m: float = 3.0
    pipe_yaw_jitter_rad: float = 0.5  # around perpendicular to the track
    # Density: one uniform draw sets both the slot occupancy and the pipe probability.
    slot_density: tuple[float, float] = (0.1, 0.65)
    pipe_prob: tuple[float, float] = (0.0, 0.75)
    keepout_m: float = 1.5


@dataclasses.dataclass(frozen=True)
class NavCommandConfig:
    """Velocity-command limits. Action in [-1, 1]^4 scales to (surge, sway, heave, yaw rate)."""

    surge_max_mps: float = 0.6
    sway_max_mps: float = 0.3
    heave_max_mps: float = 0.3
    yaw_rate_max_radps: float = 0.6


@dataclasses.dataclass(frozen=True)
class NavRewardConfig:
    w_progress_per_m: float = 2.0  # per metre of 3D distance gained towards the goal
    success_bonus: float = 25.0
    collision_penalty: float = 50.0  # also used for tilt / non-finite terminations
    w_clearance: float = 0.05  # * (1 - clearance / safe_clearance_m)^2 below safe_clearance_m
    safe_clearance_m: float = 1.0
    w_effort: float = 0.01  # mean |normalized thrust|
    w_action_rate: float = 0.02  # mean squared change of the normalized velocity command
    w_time: float = 0.01  # per control step
    w_tilt: float = 0.5
    goal_radius_m: float = 1.0
    min_cos_tilt: float = 0.5


@dataclasses.dataclass(frozen=True)
class NavConfig:
    sim: SimConfig = SimConfig(episode_duration_s=100.0)
    sonar: SonarConfig = SonarConfig()
    command: NavCommandConfig = NavCommandConfig()
    reward: NavRewardConfig = NavRewardConfig()


# Vehicle / current randomization for nav training: the plant ranges of TRAIN_RANDOMIZATION, a weaker current
# (the vehicle cruises at <= 0.6 m/s) and no thruster failures (this task studies planning, not fault tolerance).
NAV_TRAIN_VEHICLE = dataclasses.replace(
    TRAIN_RANDOMIZATION, current_const_mps=(0.0, 0.25), current_amp_mps=(0.0, 0.08), p_thruster_failure=0.0)
NAV_TRAIN_SCENE = NavSceneConfig()

# Evaluation obstacle-density levels: (slot occupancy, pipe probability).
NAV_DENSITY_LEVELS: dict[str, tuple[float, float]] = {
    "low": (0.2, 0.2),
    "medium": (0.4, 0.45),
    "high": (0.6, 0.7),
}
# Evaluation current levels are indices into DISTURBANCE_LEVELS_MPS; level 2 (0.2 m/s) is inside the
# nav training range, level 3 (0.3 m/s) is outside it.
NAV_CURRENT_LEVELS: tuple[int, ...] = (0, 2, 3)


def nav_eval_vehicle(current_level: int) -> RandomizationConfig:
    """Nominal plant, fixed current magnitude for one evaluation current level."""
    if not 0 <= current_level < len(DISTURBANCE_LEVELS_MPS):
        raise ValueError(f"current level must be in [0, {len(DISTURBANCE_LEVELS_MPS) - 1}]")
    const, amp = DISTURBANCE_LEVELS_MPS[current_level]
    return RandomizationConfig(current_const_mps=(const, const), current_amp_mps=(amp, amp))


def nav_eval_scene(density: str) -> NavSceneConfig:
    if density not in NAV_DENSITY_LEVELS:
        raise ValueError(f"unknown density '{density}', expected one of {sorted(NAV_DENSITY_LEVELS)}")
    slot, pipe = NAV_DENSITY_LEVELS[density]
    return dataclasses.replace(NAV_TRAIN_SCENE, slot_density=(slot, slot), pipe_prob=(pipe, pipe))
