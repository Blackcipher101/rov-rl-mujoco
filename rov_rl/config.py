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
