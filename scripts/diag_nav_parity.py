"""Diagnose nav sonar parity between the CPU (numpy float64) and MJX (jax float32) code paths. Needs JAX.

For one MJX reset it prints, per elevation row and beam, the CPU and JAX ranges and which primitive each ray
hit (capsule index / pipe or pillar, box index, seabed). It also probes the suspected mechanism directly:
the capsule-ray discriminant built with a matmul (`@`) at JAX's default and at "highest" precision versus the
elementwise form nav_core uses, counting rays whose hit / miss decision flips (TF32 on Ampere+ GPUs).

Run (ws1):  JAX_PLATFORMS=cuda ~/.venvs/mjrl/bin/python -u scripts/diag_nav_parity.py --seed 0
Exit code 1 if any CPU vs JAX range differs by more than --tol-m.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rov_rl import core, nav_core  # noqa: E402
from rov_rl.config import NAV_TRAIN_VEHICLE, NavConfig, SonarConfig, nav_eval_scene  # noqa: E402
from rov_rl.nav_mjx_env import NavMjxEnv  # noqa: E402

NOISELESS = dataclasses.replace(NavConfig(), sonar=dataclasses.replace(
    SonarConfig(), range_noise_std_m=0.0, range_noise_frac=0.0, dropout_prob=0.0, dvl_noise_std_mps=0.0))


def per_primitive(xp, o, d, scene: nav_core.NavScene, max_range_m: float) -> tuple[np.ndarray, list[str]]:
    """Nearest range per ray (clipped) and a label of the primitive that produced it."""
    n_cap, n_box = scene.cap_active.shape[0], scene.box_active.shape[0]
    cols, labels = [], []
    for i in range(n_cap):
        if float(scene.cap_active[i]) < 0.5:
            continue
        one = xp.zeros_like(scene.cap_active).at[i].set(1.0) if xp is jnp else np.eye(n_cap)[i]
        cols.append(np.asarray(nav_core.ray_capsules(xp, o, d, scene.cap_a_m, scene.cap_b_m, scene.cap_radius_m, one)))
        labels.append(f"{'pipe' if float(scene.cap_is_pipe[i]) > 0.5 else 'pillar'}{i}")
    for i in range(n_box):
        if float(scene.box_active[i]) < 0.5:
            continue
        one = xp.zeros_like(scene.box_active).at[i].set(1.0) if xp is jnp else np.eye(n_box)[i]
        cols.append(np.asarray(nav_core.ray_boxes(xp, o, d, scene.box_center_m, scene.box_half_m, scene.box_yaw_rad,
                                                  one)))
        labels.append(f"box{i}")
    cols.append(np.asarray(nav_core.ray_seabed(xp, o, d, scene.seabed_z_m)))
    labels.append("seabed")
    t = np.stack(cols, axis=-1).astype(np.float64)
    idx = np.argmin(t, axis=-1)
    best = np.minimum(t[np.arange(t.shape[0]), idx], max_range_m)
    names = [labels[k] if best[m] < max_range_m else "-" for m, k in enumerate(idx)]
    return best, names


def tf32_probe(o: np.ndarray, d: np.ndarray, scene: nav_core.NavScene) -> None:
    """Capsule discriminant with `@` (default / highest precision) vs elementwise, on the actual rays."""
    a, b, r = (jnp.asarray(scene.cap_a_m, jnp.float32), jnp.asarray(scene.cap_b_m, jnp.float32),
               jnp.asarray(scene.cap_radius_m, jnp.float32))
    oj, dj = jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32)
    ba = b - a
    n = ba / jnp.linalg.norm(ba, axis=-1, keepdims=True)
    oa = oj[None] - a
    oa_perp = oa - jnp.sum(oa * n, -1)[:, None] * n
    qc = jnp.sum(oa_perp**2, -1) - r**2

    def disc(qb, d_ax):
        return qb**2 - (1.0 - d_ax**2) * qc[None]

    elem = disc(jnp.sum(dj[:, None] * oa_perp[None], -1), jnp.sum(dj[:, None] * n[None], -1))
    mm_default = jax.jit(lambda: disc(dj @ oa_perp.T, dj @ n.T))()
    with jax.default_matmul_precision("highest"):
        mm_high = jax.jit(lambda: disc(dj @ oa_perp.T, dj @ n.T))()
    active = np.asarray(scene.cap_active) > 0.5
    for name, h in (("matmul default", mm_default), ("matmul highest", mm_high)):
        dh = np.abs(np.asarray(h) - np.asarray(elem))[:, active]
        flips = int(np.sum((np.asarray(h) >= 0)[:, active] != (np.asarray(elem) >= 0)[:, active]))
        print(f"[diag] tf32 probe: {name:15s} vs elementwise: max |d discriminant| = {dh.max():.4g} m^2, "
              f"hit/miss flips = {flips} of {dh.size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density", default="high")
    parser.add_argument("--tol-m", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"[diag] jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()} "
          f"default_matmul_precision={jax.config.jax_default_matmul_precision}", flush=True)
    sonar = NOISELESS.sonar
    env = NavMjxEnv(NAV_TRAIN_VEHICLE, nav_eval_scene(args.density), NOISELESS)
    state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    scene_j = state.info["params"].scene
    scene_np = nav_core.NavScene(*[np.asarray(x, dtype=np.float64) for x in scene_j])
    qpos_j = state.data.qpos
    qpos_np = np.asarray(qpos_j, dtype=np.float64)

    rot = core.quat_to_mat(np, qpos_np[3:7])
    o_np = qpos_np[:3] + rot @ np.asarray(sonar.mount_pos_m)
    d_np = nav_core.sonar_dirs_body(sonar) @ rot.T
    rot_j = core.quat_to_mat(jnp, qpos_j[3:7])
    o_j = qpos_j[:3] + jnp.sum(rot_j * jnp.asarray(sonar.mount_pos_m, jnp.float32)[None], -1)
    d_j = jnp.sum(jnp.asarray(nav_core.sonar_dirs_body(sonar), jnp.float32)[:, None, :] * rot_j[None], -1)

    t_cpu, hit_cpu = per_primitive(np, o_np, d_np, scene_np, sonar.max_range_m)
    t_jax, hit_jax = per_primitive(jnp, o_j, d_j, scene_j, sonar.max_range_m)
    n_rows, nb = len(sonar.elevation_rows_rad), sonar.num_beams
    az = np.degrees(nav_core.beam_azimuths_rad(sonar))
    print(f"[diag] seed={args.seed} obstacles: {int(scene_np.cap_active.sum())} capsules, "
          f"{int(scene_np.box_active.sum())} boxes; seabed z={float(scene_np.seabed_z_m):.2f} m", flush=True)
    for row in range(n_rows):
        sl = slice(row * nb, (row + 1) * nb)
        print(f"[diag] row {row} (el {np.degrees(sonar.elevation_rows_rad[row]):+5.1f} deg): "
              f"max |cpu - jax| = {np.max(np.abs(t_cpu[sl] - t_jax[sl])):.4g} m", flush=True)
    print("[diag] rays differing by more than tol (row, beam, azimuth, cpu range/hit, jax range/hit):", flush=True)
    bad = np.flatnonzero(np.abs(t_cpu - t_jax) > args.tol_m)
    for m in bad:
        row, beam = divmod(int(m), nb)
        where = f"row {row} beam {beam:2d} az {az[beam]:+6.1f}" if row < n_rows else f"vertical {beam}"
        print(f"[diag]   {where}: cpu {t_cpu[m]:7.3f} m ({hit_cpu[m]:9s}) jax {t_jax[m]:7.3f} m ({hit_jax[m]})",
              flush=True)

    # End-to-end: the env's per-beam (min over rows) fan, as in the observation.
    fan_cpu, vert_cpu = nav_core.true_ranges(np, qpos_np, scene_np, sonar)
    fan_jax, vert_jax = jax.jit(lambda q: nav_core.true_ranges(jnp, q, scene_j, sonar))(qpos_j)
    fan_diff = np.abs(np.asarray(fan_jax, np.float64) - fan_cpu)
    print(f"[diag] true_ranges fan: max |cpu - jax| = {fan_diff.max():.4g} m at beams "
          f"{np.flatnonzero(fan_diff > args.tol_m).tolist()}; vertical diff "
          f"{np.abs(np.asarray(vert_jax, np.float64) - vert_cpu).max():.4g} m", flush=True)
    tf32_probe(o_np, d_np, scene_np)
    ok = bad.size == 0 and fan_diff.max() <= args.tol_m
    print(f"[diag] {'PASS' if ok else 'FAIL'}: {bad.size} rays beyond {args.tol_m} m", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
