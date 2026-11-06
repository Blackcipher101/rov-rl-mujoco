"""Tune the cascade-PID gains by multiplicative coordinate search (CPU only).

Objective (lower is better), over a fixed set of tuning episodes that use a
different seed from eval.py (no tuning on the test episodes):
    mean(pos RMS, terminated episode -> FAIL_COST_M) + YAW_WEIGHT * mean(yaw RMS)
    + EFFORT_WEIGHT * mean(thrust impulse)
Episodes: station keeping and tracking at disturbance levels 1 and 3, each with the
nominal plant, the "params" randomization (mass, buoyancy, drag, thruster gain and
lag) and a single thruster failure, mirroring what the RL policy trains on. (Tuning
without failures produced low attitude gains that rolled the vehicle over when a
vertical thruster died.)
A candidate is accepted only if it lowers J by more than MIN_REL_IMPROVEMENT.

Run:  python scripts/tune_pid.py --out config/pid_gains.json
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rov_rl.config import MetricConfig, SimConfig, eval_randomization  # noqa: E402
from rov_rl.cpu_env import RovCpuEnv  # noqa: E402
from rov_rl.pid_baseline import CascadePid, PidGains  # noqa: E402
from rov_rl.rollout import PidController, episode_metrics, run_episode  # noqa: E402

TUNE_SEED = 7
TUNE_LEVELS = (1, 3)
TUNE_SCENARIOS = ("nominal", "params", "thruster_failure")
FAIL_COST_M = 2.0
YAW_WEIGHT = 0.3
EFFORT_WEIGHT = 2e-5
STEP_FACTOR = 1.5
MAX_ROUNDS = 10
MIN_REL_IMPROVEMENT = 1e-3


def _episode_cost(job: tuple[PidGains, str, int, str, int]) -> tuple[float, float, float]:
    gains, task, level, scenario, idx = job
    sim = SimConfig()
    env = RovCpuEnv(eval_randomization(task, level, scenario), sim)
    seed = [TUNE_SEED, level, idx, int(task == "track"), TUNE_SCENARIOS.index(scenario)]
    params = env.sample_params(np.random.default_rng(seed))
    log = run_episode(env, PidController(CascadePid(env.model, gains, sim.ctrl_dt_s)), params)
    m = episode_metrics(log, MetricConfig(), sim.episode_duration_s)
    rms = float(m["pos_rms_m"])
    yaw = float(m["yaw_rms_rad"])
    return (rms if math.isfinite(rms) else FAIL_COST_M, yaw if math.isfinite(yaw) else math.pi,
            float(m["thrust_impulse_ns"]))


def objective(pool: mp.pool.Pool, gains: PidGains, per_cell: int) -> float:
    jobs = [(gains, task, lvl, scen, i) for task in ("station", "track") for lvl in TUNE_LEVELS
            for scen in TUNE_SCENARIOS for i in range(per_cell)]
    res = np.array(pool.map(_episode_cost, jobs))
    return float(res[:, 0].mean() + YAW_WEIGHT * res[:, 1].mean() + EFFORT_WEIGHT * res[:, 2].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=4)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    gains = PidGains()
    names = [f.name for f in dataclasses.fields(PidGains)]
    t0 = time.monotonic()
    n_eval = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        best = objective(pool, gains, args.episodes_per_cell)
        n_eval += 1
        print(f"[tune] initial J={best:.4f}", flush=True)
        for rnd in range(MAX_ROUNDS):
            improved = False
            for name in names:
                for factor in (STEP_FACTOR, 1.0 / STEP_FACTOR):
                    cand = dataclasses.replace(gains, **{name: getattr(gains, name) * factor})
                    j = objective(pool, cand, args.episodes_per_cell)
                    n_eval += 1
                    better = j < best * (1.0 - MIN_REL_IMPROVEMENT)
                    if better:
                        gains, best, improved = cand, j, True
                    print(f"[tune] round={rnd} eval={n_eval} {name}*{factor:.3f} -> {getattr(cand, name):.4f} "
                          f"J={j:.4f} best={best:.4f} {'ACCEPT' if better else ''} "
                          f"t={time.monotonic() - t0:.0f}s", flush=True)
                    if better:
                        break
            if not improved:
                break
    note = (f"Tuned by scripts/tune_pid.py: J={best:.4f} after {n_eval} objective evaluations, "
            f"seed {TUNE_SEED}, levels {TUNE_LEVELS}, scenarios {TUNE_SCENARIOS}, "
            f"{args.episodes_per_cell} episodes per task/level/scenario.")
    gains.save(args.out, note)
    print(f"[tune] {note}\n[tune] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
