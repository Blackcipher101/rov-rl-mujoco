"""Evaluate PID and/or RL controllers on identical CPU-MuJoCo episodes.

Writes <out-dir>/metrics.json (per episode + aggregates), <out-dir>/results.md
(RL vs PID table per disturbance level and robustness scenario) and, unless
--no-video, chase-camera MP4s under <out-dir>/videos/ (needs an OpenGL backend,
headless: MUJOCO_GL=egl, plus imageio + imageio-ffmpeg).

Both controllers see exactly the same sampled episodes (paired comparison).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np

from rov_rl import core
from rov_rl.config import DISTURBANCE_LEVELS_MPS, MetricConfig, SimConfig, eval_randomization
from rov_rl.cpu_env import RovCpuEnv
from rov_rl.pid_baseline import DEFAULT_GAINS_PATH, CascadePid, PidGains
from rov_rl.rollout import Controller, PidController, PolicyController, aggregate, episode_metrics, run_episode

TASKS = ("station", "track")
ROBUSTNESS_SCENARIOS = ("params", "thruster_failure", "combined")
TALLY_EVERY = 10


@dataclasses.dataclass(frozen=True)
class Cell:
    task: str
    level: int
    scenario: str

    @property
    def name(self) -> str:
        return f"{self.task}_{self.scenario}_L{self.level}"


def build_cells(tasks: list[str], levels: list[int], robustness_level: int, scenarios: list[str]) -> list[Cell]:
    cells = [Cell(t, lvl, "nominal") for t in tasks for lvl in levels]
    cells += [Cell(t, robustness_level, s) for t in tasks for s in scenarios]
    return cells


def cell_params(env: RovCpuEnv, cell: Cell, n: int, seed: int) -> list[core.EpisodeParams]:
    """Deterministic episode list per cell, independent of the controller."""
    task_idx = TASKS.index(cell.task)
    scen_idx = (("nominal",) + ROBUSTNESS_SCENARIOS).index(cell.scenario)
    return [
        env.sample_params(np.random.default_rng([seed, task_idx, cell.level, scen_idx, i]))
        for i in range(n)
    ]


def json_safe(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    return x


def fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def results_table(rows: list[dict[str, Any]], controllers: list[str]) -> str:
    header = ("| task | scenario | level (current const/amp m/s) | controller | pos RMS [m] | pos max [m] "
              "| settling [s] | thrust impulse [N s] | success | terminated |")
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        const, amp = DISTURBANCE_LEVELS_MPS[r["level"]]
        for c in controllers:
            a = r["aggregate"].get(c)
            if a is None:
                continue
            lines.append(
                f"| {r['task']} | {r['scenario']} | {r['level']} ({const:.2f}/{amp:.2f}) | {c} | "
                f"{fmt(a['pos_rms_m'])} | {fmt(a['pos_max_m'])} | {fmt(a['settling_time_s'], 1)} | "
                f"{fmt(a['thrust_impulse_ns'], 0)} | {a['success_rate']:.2f} | {a['termination_rate']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controllers", default="pid,rl", help="comma list of: pid, rl")
    parser.add_argument("--policy-dir", type=pathlib.Path, help="train.py output <run>/policy (needed for rl)")
    parser.add_argument("--pid-gains", type=pathlib.Path, default=DEFAULT_GAINS_PATH)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per cell")
    parser.add_argument("--tasks", default="station,track")
    parser.add_argument("--levels", default=",".join(str(i) for i in range(len(DISTURBANCE_LEVELS_MPS))))
    parser.add_argument("--robustness-level", type=int, default=2)
    parser.add_argument("--scenarios", default=",".join(ROBUSTNESS_SCENARIOS), help="robustness scenarios ('' = none)")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-cells",
                        default="station_nominal_L2,track_nominal_L2,station_nominal_L4,track_combined_L2",
                        help="cell names to film (first episode of each)")
    args = parser.parse_args()

    controllers = [c for c in args.controllers.split(",") if c]
    if not set(controllers) <= {"pid", "rl"} or not controllers:
        parser.error("--controllers must be a comma list of pid, rl")
    if "rl" in controllers and args.policy_dir is None:
        parser.error("--policy-dir is required for the rl controller")
    tasks = [t for t in args.tasks.split(",") if t]
    levels = [int(x) for x in args.levels.split(",") if x]
    scenarios = [s for s in args.scenarios.split(",") if s]
    cells = build_cells(tasks, levels, args.robustness_level, scenarios)
    video_cells = set() if args.no_video else set(args.video_cells.split(","))

    sim = SimConfig()
    metric_cfg = MetricConfig()
    probe = RovCpuEnv(eval_randomization("station", 0), sim)
    impls: dict[str, Controller] = {}
    if "pid" in controllers:
        impls["pid"] = PidController(CascadePid(probe.model, PidGains.load(args.pid_gains), sim.ctrl_dt_s))
    if "rl" in controllers:
        from rov_rl.policy_io import load_policy  # needs jax + brax

        impls["rl"] = PolicyController(load_policy(args.policy_dir))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(cells) * len(controllers) * args.episodes
    done = 0
    walls: list[float] = []
    successes = 0
    rows: list[dict[str, Any]] = []
    t_run = time.monotonic()
    for cell in cells:
        env = RovCpuEnv(eval_randomization(cell.task, cell.level, cell.scenario), sim)
        params_list = cell_params(env, cell, args.episodes, args.seed)
        row: dict[str, Any] = {"task": cell.task, "level": cell.level, "scenario": cell.scenario,
                               "aggregate": {}, "episodes": {}}
        for name in controllers:
            ctrl = impls[name]
            per_ep = []
            for i, params in enumerate(params_list):
                t0 = time.monotonic()
                log = run_episode(env, ctrl, params)
                m = episode_metrics(log, metric_cfg, sim.episode_duration_s)
                wall = time.monotonic() - t0
                per_ep.append(m)
                done += 1
                walls.append(wall)
                successes += int(m["success"])
                print(
                    f"[eval] {done:>5d}/{total} ctrl={name} cell={cell.name} ep={i + 1}/{args.episodes} "
                    f"rms={fmt(m['pos_rms_m'])}m max={fmt(m['pos_max_m'])}m settle={fmt(m['settling_time_s'], 1)}s "
                    f"impulse={m['thrust_impulse_ns']:.0f}Ns success={int(m['success'])} "
                    f"term={int(m['terminated'])} wall={wall:.2f}s",
                    flush=True,
                )
                if done % TALLY_EVERY == 0 or done == total:
                    med = statistics.median(walls)
                    print(
                        f"[eval] tally {done}/{total} success_so_far={successes / done:.3f} "
                        f"median_wall={med:.2f}s eta={(total - done) * med / 60.0:.1f}min",
                        flush=True,
                    )
                if i == 0 and cell.name in video_cells:
                    from rov_rl.render import render_episode, write_video

                    path = args.out_dir / "videos" / f"{cell.name}_{name}.mp4"
                    write_video(render_episode(log, params, sim.ctrl_dt_s), path, fps=1.0 / sim.ctrl_dt_s)
                    print(f"[eval] video -> {path}", flush=True)
            row["aggregate"][name] = aggregate(per_ep)
            row["episodes"][name] = per_ep
        rows.append(row)
        # Rewrite outputs after every cell so partial results survive interruption.
        meta = {"controllers": controllers, "episodes_per_cell": args.episodes, "seed": args.seed,
                "policy_dir": str(args.policy_dir) if args.policy_dir else None,
                "pid_gains": str(args.pid_gains), "metric_config": dataclasses.asdict(metric_cfg),
                "disturbance_levels_mps": DISTURBANCE_LEVELS_MPS,
                "elapsed_s": round(time.monotonic() - t_run, 1)}
        (args.out_dir / "metrics.json").write_text(
            json.dumps(json_safe({"meta": meta, "cells": rows}), indent=1) + "\n", encoding="utf-8")
        (args.out_dir / "results.md").write_text(results_table(rows, controllers), encoding="utf-8")
    print(f"[eval] finished {total} episodes in {(time.monotonic() - t_run) / 60.0:.1f} min", flush=True)
    print(results_table(rows, controllers), flush=True)


if __name__ == "__main__":
    sys.exit(main())
