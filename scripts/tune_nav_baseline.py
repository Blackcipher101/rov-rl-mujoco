"""Tune the VFH+ nav baseline by multiplicative coordinate search (CPU only).

Objective (lower is better), mean over a fixed set of tuning episodes whose seed (TUNE_SEED) differs from
eval_nav.py's, so the baseline is never tuned on test episodes:
    success   -> time_to_goal_s / (straight_line_m / REFERENCE_SPEED_MPS)   (~1-2)
    collision -> COLLISION_COST
    timeout / tilt failure -> TIMEOUT_COST
Episodes: every obstacle-density level at current levels TUNE_CURRENT_LEVELS. Only the parameters in
TUNED_PARAMS are searched; candidates that violate VfhParams.validate() are skipped. A candidate is accepted
only if it lowers J by more than MIN_REL_IMPROVEMENT.

--decay-only (nav2): start from --init and only choose logodds_decay_per_s from DECAY_GRID, on the same
tuning episodes with static AND moving obstacles (moving fields use an extra seed component, as in eval_nav.py).

Run:  python scripts/tune_nav_baseline.py --out config/nav_baseline.json
      python scripts/tune_nav_baseline.py --decay-only --init config/nav_baseline.json \
          --out config/nav_baseline_decay.json
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rov_rl.config import (  # noqa: E402
    NAV_DENSITY_LEVELS,
    NavConfig,
    nav_eval_scene,
    nav_eval_scene_moving,
    nav_eval_vehicle,
)
from rov_rl.nav_baseline import VfhParams, VfhPlanner  # noqa: E402
from rov_rl.nav_cpu_env import NavCpuEnv  # noqa: E402
from rov_rl.nav_rollout import VfhController, nav_episode_metrics, run_nav_episode  # noqa: E402

TUNE_SEED = 7
TUNE_CURRENT_LEVELS = (1, 3)
REFERENCE_SPEED_MPS = 0.5
COLLISION_COST = 10.0
TIMEOUT_COST = 5.0
TUNED_PARAMS = ("safety_margin_m", "hist_high", "window_radius_m", "cruise_speed_mps", "mu_goal", "k_yaw_ps",
                "stop_range_m", "slow_range_m", "occupied_logodds")
STEP_FACTOR = 1.4
MAX_ROUNDS = 4
MIN_REL_IMPROVEMENT = 1e-3
DECAY_GRID_PER_S = (0.0, 0.05, 0.1, 0.2, 0.4)


def _episode_cost(job: tuple[VfhParams, str, int, int, bool]) -> tuple[float, str]:
    params, density, level, idx, moving = job
    cfg = NavConfig()
    scene = nav_eval_scene_moving(density) if moving else nav_eval_scene(density)
    env = NavCpuEnv(nav_eval_vehicle(level), scene, cfg)
    seed = [TUNE_SEED, sorted(NAV_DENSITY_LEVELS).index(density), level, idx] + ([1] if moving else [])
    episode = env.sample_params(np.random.default_rng(seed))
    log = run_nav_episode(env, VfhController(VfhPlanner(params, cfg.sonar, cfg.command, cfg.sim.ctrl_dt_s)), episode,
                          noise_seed=int(np.random.default_rng(seed).integers(2**31)))
    m = nav_episode_metrics(log, episode)
    if m["outcome"] == "success":
        return float(m["time_to_goal_s"]) / (float(m["straight_line_m"]) / REFERENCE_SPEED_MPS), "success"
    if m["outcome"] == "collision":
        return COLLISION_COST, "collision"
    return TIMEOUT_COST, str(m["outcome"])


def objective(pool: mp.pool.Pool, params: VfhParams, per_cell: int,
              moving: tuple[bool, ...] = (False,)) -> tuple[float, dict[str, int]]:
    jobs = [(params, d, lvl, i, mv) for mv in moving for d in NAV_DENSITY_LEVELS for lvl in TUNE_CURRENT_LEVELS
            for i in range(per_cell)]
    res = pool.map(_episode_cost, jobs)
    counts: dict[str, int] = {}
    for _, outcome in res:
        counts[outcome] = counts.get(outcome, 0) + 1
    return float(np.mean([c for c, _ in res])), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--init", type=pathlib.Path, help="start from this params file (default: VfhParams())")
    parser.add_argument("--episodes-per-cell", type=int, default=4)
    parser.add_argument("--decay-only", action="store_true", help="only choose logodds_decay_per_s (nav2)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    params = VfhParams.load(args.init) if args.init else VfhParams()
    if args.decay_only:
        tune_decay(params, args)
        return
    t0 = time.monotonic()
    n_eval = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        best, counts = objective(pool, params, args.episodes_per_cell)
        n_eval += 1
        print(f"[tune] initial J={best:.4f} outcomes={counts} t={time.monotonic() - t0:.0f}s", flush=True)
        for rnd in range(MAX_ROUNDS):
            improved = False
            for name in TUNED_PARAMS:
                for factor in (STEP_FACTOR, 1.0 / STEP_FACTOR):
                    cand = dataclasses.replace(params, **{name: getattr(params, name) * factor})
                    try:
                        cand.validate()
                    except ValueError as exc:
                        print(f"[tune] skip {name}*{factor:.3f}: {exc}", flush=True)
                        continue
                    j, counts = objective(pool, cand, args.episodes_per_cell)
                    n_eval += 1
                    better = j < best * (1.0 - MIN_REL_IMPROVEMENT)
                    if better:
                        params, best, improved = cand, j, True
                    print(f"[tune] round={rnd} eval={n_eval} {name}*{factor:.3f} -> {getattr(cand, name):.4f} "
                          f"J={j:.4f} best={best:.4f} outcomes={counts} {'ACCEPT' if better else ''} "
                          f"t={time.monotonic() - t0:.0f}s", flush=True)
                    if better:
                        break
            if not improved:
                break
    note = (f"Tuned by scripts/tune_nav_baseline.py: J={best:.4f} after {n_eval} objective evaluations, "
            f"seed {TUNE_SEED}, current levels {TUNE_CURRENT_LEVELS}, densities {sorted(NAV_DENSITY_LEVELS)}, "
            f"{args.episodes_per_cell} episodes per density/level.")
    params.save(args.out, note)
    print(f"[tune] {note}\n[tune] wrote {args.out}", flush=True)


def tune_decay(params: VfhParams, args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for decay in DECAY_GRID_PER_S:
            cand = dataclasses.replace(params, logodds_decay_per_s=decay)
            j, counts = objective(pool, cand, args.episodes_per_cell, moving=(False, True))
            results.append((j, decay))
            print(f"[tune] decay={decay:.3f}/s J={j:.4f} outcomes={counts} t={time.monotonic() - t0:.0f}s", flush=True)
    best_j, best_decay = min(results)
    note = (f"Tuned by scripts/tune_nav_baseline.py --decay-only from {args.init}: logodds_decay_per_s chosen from "
            f"{DECAY_GRID_PER_S} (J={best_j:.4f}; all: {[(d, round(j, 4)) for j, d in results]}), seed {TUNE_SEED}, "
            f"current levels {TUNE_CURRENT_LEVELS}, densities {sorted(NAV_DENSITY_LEVELS)}, static + moving, "
            f"{args.episodes_per_cell} episodes per density/level/scenario. All other parameters unchanged.")
    dataclasses.replace(params, logodds_decay_per_s=best_decay).save(args.out, note)
    print(f"[tune] {note}\n[tune] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
