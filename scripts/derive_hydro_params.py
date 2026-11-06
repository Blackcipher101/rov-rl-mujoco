"""Derive the MJCF inertial / fluid numbers that reproduce the identified damping.

Why this exists
---------------
The MJX JAX backend implements only MuJoCo's *inertia-box* fluid model
(mujoco-mjx 3.17.0, ``_src/passive.py::_inertia_box_fluid_model``); it ignores
per-geom ``fluidshape="ellipsoid"`` without raising. To keep CPU MuJoCo and MJX
physically identical we use the inertia-box model on both.

In that model every body with mass m and principal inertia I gets drag from its
"equivalent inertia box" with edge lengths b_i = sqrt(6 (I_j + I_k - I_i) / m):

    quadratic:  F_i = -0.5 * rho * b_j * b_k * |v_i| v_i
    linear:     F_i = -3 * pi * mu * mean(b) * v_i
                T_i = -pi * mu * mean(b)^3 * w_i
    angular:    T_i = -(rho / 64) * b_i * (b_j^4 + b_k^4) * |w_i| w_i

The vehicle is split into a "hull" body (carries almost all mass and the CAD
inertia) and a light, welded "hydro proxy" child body whose inertia box is
sized so that the *sum* of both boxes reproduces the identified quadratic
translational damping of [vB22] per axis. The proxy's own inertia is
subtracted from the hull so the total rigid-body inertia stays equal to CAD.
The viscosity is an *effective* (eddy) viscosity chosen to reproduce the
identified linear surge damping X_u; it is not water's molecular viscosity
(~0.001 Pa s), whose Stokes term is orders of magnitude too small for an
open-frame ROV.

Run:  python scripts/derive_hydro_params.py
The printed values are pasted into rov_rl/assets/bluerov2_heavy.xml and are
checked by tests/test_fluid.py::test_damping_matches_identified.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rov_rl.vehicle import VEHICLE, WATER_DENSITY_KGPM3  # noqa: E402

PROXY_MASS_KG = 0.25
FIXED_POINT_ITERS = 50
FIXED_POINT_TOL = 1e-12


def box_from_inertia(mass_kg: float, inertia: np.ndarray) -> np.ndarray:
    s = np.array([
        inertia[1] + inertia[2] - inertia[0],
        inertia[0] + inertia[2] - inertia[1],
        inertia[0] + inertia[1] - inertia[2],
    ])
    if np.any(s <= 0.0):
        raise ValueError(f"inertia violates triangle inequality: {inertia}")
    return np.sqrt(6.0 * s / mass_kg)


def inertia_from_box(mass_kg: float, box_m: np.ndarray) -> np.ndarray:
    b2 = box_m**2
    return mass_kg / 12.0 * np.array([b2[1] + b2[2], b2[0] + b2[2], b2[0] + b2[1]])


def quad_coeffs(box_m: np.ndarray, rho: float) -> np.ndarray:
    return 0.5 * rho * np.array([box_m[1] * box_m[2], box_m[0] * box_m[2], box_m[0] * box_m[1]])


def box_from_quad_targets(target: np.ndarray, rho: float) -> np.ndarray:
    """Invert quad_coeffs: products p_x = b_y b_z etc."""
    if np.any(target <= 0.0):
        raise ValueError(f"hull alone already exceeds identified drag: residual {target}")
    p = 2.0 * target / rho
    bx = math.sqrt(p[1] * p[2] / p[0])
    by = p[2] / bx
    bz = p[0] / by
    return np.array([bx, by, bz])


def derive() -> dict[str, np.ndarray | float]:
    rho = WATER_DENSITY_KGPM3
    total_inertia = np.array(VEHICLE.inertia_kgm2)
    hull_mass = VEHICLE.mass_kg - PROXY_MASS_KG
    target_quad = np.array(VEHICLE.quadratic_damping[:3])
    proxy_inertia = np.zeros(3)
    for _ in range(FIXED_POINT_ITERS):
        hull_inertia = total_inertia - proxy_inertia
        hull_box = box_from_inertia(hull_mass, hull_inertia)
        proxy_box = box_from_quad_targets(target_quad - quad_coeffs(hull_box, rho), rho)
        new_proxy_inertia = inertia_from_box(PROXY_MASS_KG, proxy_box)
        if np.max(np.abs(new_proxy_inertia - proxy_inertia)) < FIXED_POINT_TOL:
            proxy_inertia = new_proxy_inertia
            break
        proxy_inertia = new_proxy_inertia
    else:
        raise RuntimeError("fixed point did not converge")
    hull_inertia = total_inertia - proxy_inertia
    hull_box = box_from_inertia(hull_mass, hull_inertia)
    mean_sizes = hull_box.mean() + proxy_box.mean()
    viscosity = VEHICLE.linear_damping[0] / (3.0 * math.pi * mean_sizes)
    ang_lin = math.pi * viscosity * (hull_box.mean() ** 3 + proxy_box.mean() ** 3)

    def ang_quad(b: np.ndarray) -> np.ndarray:
        return rho / 64.0 * np.array([
            b[0] * (b[1] ** 4 + b[2] ** 4),
            b[1] * (b[0] ** 4 + b[2] ** 4),
            b[2] * (b[0] ** 4 + b[1] ** 4),
        ])

    return {
        "hull_mass_kg": hull_mass,
        "hull_inertia_kgm2": hull_inertia,
        "hull_box_m": hull_box,
        "proxy_mass_kg": PROXY_MASS_KG,
        "proxy_inertia_kgm2": proxy_inertia,
        "proxy_box_m": proxy_box,
        "viscosity_pas": viscosity,
        "quad_translational": quad_coeffs(hull_box, rho) + quad_coeffs(proxy_box, rho),
        "linear_translational": np.full(3, 3.0 * math.pi * viscosity * mean_sizes),
        "linear_rotational": np.full(3, ang_lin),
        "quad_rotational": ang_quad(hull_box) + ang_quad(proxy_box),
    }


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    for key, value in derive().items():
        print(f"{key:24s} {value}")


if __name__ == "__main__":
    main()
