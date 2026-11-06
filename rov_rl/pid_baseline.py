"""Cascade PID baseline with pseudo-inverse thruster allocation.

Structure (all gains in config/pid_gains.json, tuned by scripts/tune_pid.py):

  position PI (world)  ->  velocity command (+ reference feed-forward, saturated)
  velocity P  (body)   ->  force = M * Kp_vel * v_err + nominal drag feed-forward
  yaw P (+ reference yaw-rate feed-forward) -> yaw-rate P -> torque_z
  roll/pitch PD        ->  torque_x, torque_y  (level attitude)
  + nominal net-buoyancy feed-forward
  wrench (6) -> thrusts (8) via Moore-Penrose pseudo-inverse of the allocation
  matrix, uniformly scaled down if any thruster exceeds its limit.

Anti-windup: conditional integration (the integrator is frozen on any step in
which the allocation saturated) plus a hard clamp. The controller knows only
the nominal vehicle model; it is not told about currents, plant randomization
or thruster failures (same information as the RL policy).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import mujoco
import numpy as np

from rov_rl import core
from rov_rl.vehicle import (
    DEFAULT_BUOYANCY_RATIO,
    GRAVITY_MPS2,
    NUM_THRUSTERS,
    VEHICLE,
    allocation_matrix,
)

DEFAULT_GAINS_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "pid_gains.json"


@dataclasses.dataclass(frozen=True)
class PidGains:
    kp_pos_ps: float = 0.8  # [1/s]
    ki_pos_ps2: float = 0.1  # [1/s^2]
    integral_limit_ms: float = 5.0  # [m s] per axis
    v_max_mps: float = 0.6
    kp_vel_ps: float = 3.0  # [1/s], multiplied by rigid-body mass
    kp_yaw_ps: float = 1.5  # [1/s]
    yaw_rate_max_radps: float = 1.0
    kp_yaw_rate_ps: float = 4.0  # [1/s], multiplied by Izz
    kp_att_ps2: float = 16.0  # [1/s^2], multiplied by Ixx / Iyy
    kd_att_ps: float = 6.0  # [1/s], multiplied by Ixx / Iyy

    @staticmethod
    def load(path: pathlib.Path = DEFAULT_GAINS_PATH) -> "PidGains":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(PidGains)}
        unknown = set(raw["gains"]) - known
        if unknown:
            raise ValueError(f"unknown PID gain keys in {path}: {sorted(unknown)}")
        gains = PidGains(**raw["gains"])
        for f in dataclasses.fields(gains):
            value = getattr(gains, f.name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"PID gain {f.name}={value} must be finite and >= 0")
        return gains

    def save(self, path: pathlib.Path, note: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"note": note, "gains": dataclasses.asdict(self)}, f, indent=2)
            f.write("\n")


def allocate(tam_pinv: np.ndarray, wrench_body: np.ndarray, thrust_max_n: float) -> tuple[np.ndarray, bool]:
    """Wrench -> thrusts [N]; uniform scaling keeps the wrench direction on saturation."""
    thrust = tam_pinv @ wrench_body
    peak = float(np.max(np.abs(thrust)))
    if peak > thrust_max_n:
        return thrust * (thrust_max_n / peak), True
    return thrust, False


class CascadePid:
    def __init__(self, model: mujoco.MjModel, gains: PidGains, ctrl_dt_s: float) -> None:
        self.gains = gains
        self.dt_s = ctrl_dt_s
        tam = allocation_matrix(model)
        if np.linalg.matrix_rank(tam) < 6:
            raise ValueError("thruster allocation matrix is rank deficient")
        self.tam_pinv = np.linalg.pinv(tam)
        self.mass_kg = VEHICLE.mass_kg
        self.inertia = np.array(VEHICLE.inertia_kgm2)
        self.lin_damp = np.array(VEHICLE.linear_damping[:3])
        self.quad_damp = np.array(VEHICLE.quadratic_damping[:3])
        self.net_buoyancy_n = (DEFAULT_BUOYANCY_RATIO - 1.0) * VEHICLE.mass_kg * GRAVITY_MPS2
        self.integral = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)

    def wrench(self, qpos: np.ndarray, qvel: np.ndarray, ref: core.Reference) -> tuple[np.ndarray, np.ndarray]:
        """Desired body wrench and the position error used for integration."""
        g = self.gains
        rot = core.quat_to_mat(np, qpos[3:7])
        err = ref.pos_m - qpos[:3]
        v_cmd = ref.vel_mps + g.kp_pos_ps * err + g.ki_pos_ps2 * self.integral
        speed = float(np.linalg.norm(v_cmd))
        if speed > g.v_max_mps:
            v_cmd *= g.v_max_mps / speed
        v_cmd_body = rot.T @ v_cmd
        v_err_body = v_cmd_body - rot.T @ qvel[:3]
        drag_ff = self.lin_damp * v_cmd_body + self.quad_damp * np.abs(v_cmd_body) * v_cmd_body
        force = self.mass_kg * g.kp_vel_ps * v_err_body + drag_ff
        force += rot.T @ np.array([0.0, 0.0, -self.net_buoyancy_n])

        yaw_err = float(core.wrap_angle(np, ref.yaw_rad - core.yaw_of(np, rot)))
        r_cmd = float(
            np.clip(ref.yaw_rate_radps + g.kp_yaw_ps * yaw_err, -g.yaw_rate_max_radps, g.yaw_rate_max_radps)
        )
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        pitch = float(-np.arcsin(np.clip(rot[2, 0], -1.0, 1.0)))
        torque = np.array([
            -self.inertia[0] * (g.kp_att_ps2 * roll + g.kd_att_ps * qvel[3]),
            -self.inertia[1] * (g.kp_att_ps2 * pitch + g.kd_att_ps * qvel[4]),
            self.inertia[2] * g.kp_yaw_rate_ps * (r_cmd - qvel[5]),
        ])
        return np.concatenate([force, torque]), err

    def act(self, qpos: np.ndarray, qvel: np.ndarray, ref: core.Reference) -> np.ndarray:
        """Normalized thruster command in [-1, 1]^8 (same action space as the policy)."""
        wrench, err = self.wrench(qpos, qvel, ref)
        thrust, saturated = allocate(self.tam_pinv, wrench, VEHICLE.thrust_max_n)
        if not saturated:
            lim = self.gains.integral_limit_ms
            self.integral = np.clip(self.integral + err * self.dt_s, -lim, lim)
        action = thrust / VEHICLE.thrust_max_n
        if action.shape != (NUM_THRUSTERS,) or not np.all(np.isfinite(action)):
            raise FloatingPointError(f"PID produced invalid thruster command {action}")
        return action
