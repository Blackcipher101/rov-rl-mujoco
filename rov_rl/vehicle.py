"""BlueROV2-Heavy-like vehicle parameters and model loading.

All numbers below come from a single public source unless stated otherwise:

    [vB22] M. von Benzon, F. F. Sorensen, E. Uth, J. Jouffroy, J. Liniger,
           S. Pedersen, "An Open-Source Benchmark Simulator: Control of a
           BlueROV2 Underwater Robot", J. Mar. Sci. Eng. 10(12):1898, 2022.
           doi:10.3390/jmse10121898  (Table A1, Table 4, Table 5, Eqs. 18-19).

Frame conventions
-----------------
[vB22] uses the marine NED body frame (x forward, y starboard, z down).
This package uses MuJoCo's convention: world z up, body frame FLU
(x forward, y port/left, z up). A NED body vector (x, y, z) maps to FLU as
(x, -y, -z). All values stored here are already converted to FLU.

Discrepancy flagged
-------------------
Table A1 of [vB22] lists the horizontal thruster orientation angles as
beta = {0, pi/2, 3*pi/4, pi}. beta = 3*pi/4 yields a pure-sway thruster that
breaks the symmetric vectored layout shown in their Fig. 3; we use
beta = 3*pi/2, which restores the standard 45-degree vectored "X" layout.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import mujoco
import numpy as np

ASSET_DIR = pathlib.Path(__file__).resolve().parent / "assets"
MJCF_PATH = ASSET_DIR / "bluerov2_heavy.xml"

NUM_THRUSTERS = 8
GRAVITY_MPS2 = 9.81
WATER_DENSITY_KGPM3 = 1000.0


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ned_to_flu(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[1], -v[2]])


def _thruster_geometry_flu() -> tuple[np.ndarray, np.ndarray]:
    """Thruster positions [m] and unit thrust axes, FLU body frame, [vB22] Table A1."""
    pos_ned: list[np.ndarray] = []
    axis_ned: list[np.ndarray] = []
    horizontal_pos = np.array([0.156, 0.111, 0.085])
    horizontal_axis = np.array([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0])
    alphas_rad = (0.0, 5.05, 1.91, math.pi)
    betas_rad = (0.0, math.pi / 2.0, 3.0 * math.pi / 2.0, math.pi)  # see module docstring
    for alpha, beta in zip(alphas_rad, betas_rad):
        pos_ned.append(_rot_z(alpha) @ horizontal_pos)
        axis_ned.append(_rot_z(beta) @ horizontal_axis)
    vertical_pos = np.array([0.120, 0.218, 0.0])
    for gamma in (0.0, 4.15, 1.01, math.pi):
        pos_ned.append(_rot_z(gamma) @ vertical_pos)
        axis_ned.append(np.array([0.0, 0.0, -1.0]))
    pos = np.stack([_ned_to_flu(p) for p in pos_ned])
    axis = np.stack([_ned_to_flu(a) for a in axis_ned])
    return pos, axis


@dataclasses.dataclass(frozen=True)
class VehicleParams:
    """Published BlueROV2 Heavy parameters, SI units, FLU body frame."""

    mass_kg: float = 13.5
    # Principal inertia about the CoG [kg m^2] (CAD values, [vB22] Table A1).
    inertia_kgm2: tuple[float, float, float] = (0.26, 0.23, 0.37)
    # Displaced volume [m^3]; [vB22] gives 0.0134 m^3, i.e. B/W = 0.993
    # (slightly heavy). We trim the vehicle slightly positive by default
    # (see DEFAULT_BUOYANCY_RATIO), as is common practice for ROV operation.
    displaced_volume_m3: float = 0.0134
    # Centre of buoyancy relative to the CoG, FLU [m]; [vB22] NED (0, 0, -0.01).
    cob_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.01)
    # Identified damping, [vB22] Table A1 (Fossen notation, magnitudes).
    linear_damping: tuple[float, ...] = (13.7, 0.0, 33.0, 0.0, 0.8, 0.0)
    quadratic_damping: tuple[float, ...] = (141.0, 217.0, 190.0, 1.19, 0.47, 1.5)
    # Added mass [kg, kg m^2]; NOT simulated (see README "Model limitations").
    added_mass: tuple[float, ...] = (6.36, 7.12, 18.68, 0.189, 0.135, 0.222)
    # Per-thruster force limit [N]: F(V=+-1) of the regression in [vB22] Eq. 18.
    thrust_max_n: float = 30.4
    # First-order thruster lag [s]: moment-matched (mean delay) equivalent of
    # the identified 3rd-order transfer function [vB22] Eq. 19:
    #   tau = a1/a0 - b1/b0 = 9258/108700 - 6136/108700 = 0.0287 s.
    thruster_tau_s: float = 0.0287


VEHICLE = VehicleParams()
DEFAULT_BUOYANCY_RATIO = 1.01  # B / W; ~ +1.3 N net buoyancy for 13.5 kg.
THRUSTER_POS_M, THRUSTER_AXIS = _thruster_geometry_flu()


def load_model() -> mujoco.MjModel:
    """Compile the packaged MJCF."""
    return mujoco.MjModel.from_xml_path(str(MJCF_PATH))


def thruster_site_names() -> list[str]:
    return [f"thruster{i + 1}" for i in range(NUM_THRUSTERS)]


def allocation_matrix(model: mujoco.MjModel) -> np.ndarray:
    """6x8 thruster allocation matrix B (body wrench = B @ thrusts), from the model.

    Read from the compiled site poses (not from VehicleParams), so it always
    matches what the simulator actually applies. Rows: [Fx Fy Fz Tx Ty Tz] in
    the FLU body frame about the vehicle body origin (== CoG).
    """
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rov")
    if root_id < 0:
        raise ValueError("MJCF has no body named 'rov'")
    tam = np.zeros((6, NUM_THRUSTERS))
    for j, name in enumerate(thruster_site_names()):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"MJCF has no site named '{name}'")
        if model.site_bodyid[sid] != root_id:
            raise ValueError(f"site '{name}' must be attached to body 'rov'")
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[sid])
        axis = rot.reshape(3, 3)[:, 2]  # actuator gear is along site +z
        pos = model.site_pos[sid] - model.body_ipos[root_id]
        tam[:3, j] = axis
        tam[3:, j] = np.cross(pos, axis)
    return tam
