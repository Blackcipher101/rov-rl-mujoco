"""Evaluate the nav task (local planning / obstacle avoidance): RL policy vs VFH+ baseline on identical episodes.

Both controllers see the same sampled obstacle fields, currents and sensor-noise sequences (paired comparison),
in CPU MuJoCo. Cells: obstacle-density level x current level. Writes
  <out-dir>/metrics.json   per-episode metrics + aggregates (rewritten after every cell)
  <out-dir>/results.md     table per cell and controller
  <out-dir>/videos/        chase-camera MP4s with the sonar fan, per controller and side by side (rl | baseline)
  <out-dir>/paths_<cell>.png overhead plot of one field with every controller's path
Videos need an OpenGL backend (headless: MUJOCO_GL=egl) and imageio + imageio-ffmpeg; the plot needs matplotlib.
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

from rov_rl import nav_core
from rov_rl.config import (
    DISTURBANCE_LEVELS_MPS,
    NAV_CURRENT_LEVELS,
    NAV_DENSITY_LEVELS,
    NavConfig,
    nav_eval_scene,
    nav_eval_vehicle,
)
from rov_rl.nav_baseline import DEFAULT_VFH_PATH, VfhParams, VfhPlanner
from rov_rl.nav_cpu_env import NavCpuEnv
from rov_rl.nav_rollout import (
    NavController,
    NavEpisodeLog,
    NavPolicyController,
    VfhController,
    aggregate_nav,
    nav_episode_metrics,
    run_nav_episode,
)

CONTROLLERS = ("baseline", "rl")
DENSITY_ORDER = ("low", "medium", "high")
TALLY_EVERY = 10


@dataclasses.dataclass(frozen=True)
class NavCell:
    density: str
    level: int

    @property
    def name(self) -> str:
        return f"{self.density}_L{self.level}"


def cell_episodes(env: NavCpuEnv, cell: NavCell, n: int, seed: int) -> list[tuple[nav_core.NavEpisode, int]]:
    """Deterministic (episode params, noise seed) list per cell, independent of the controller."""
    out = []
    for i in range(n):
        key = [seed, DENSITY_ORDER.index(cell.density), cell.level, i]
        noise_seed = int(np.random.default_rng(key + [1]).integers(2**31))
        out.append((env.sample_params(np.random.default_rng(key)), noise_seed))
    return out


def json_safe(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    return x


def fmt(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def results_table(rows: list[dict[str, Any]], controllers: list[str]) -> str:
    header = ("| density | current (const/amp m/s) | controller | success | collision | timeout | time to goal [s] "
              "| path ratio | min clearance mean / worst [m] | thrust impulse [N s] |")
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        if r["density"] == "all":
            current = "all"
        else:
            const, amp = DISTURBANCE_LEVELS_MPS[r["level"]]
            current = f"{r['level']} ({const:.2f}/{amp:.2f})"
        for c in controllers:
            a = r["aggregate"].get(c)
            if a is None:
                continue
            lines.append(
                f"| {r['density']} | {current} | {c} | {a['success_rate']:.2f} | {a['collision_rate']:.2f} | "
                f"{a['timeout_rate'] + a['failure_rate']:.2f} | {fmt(a['time_to_goal_s'], 1)} | "
                f"{fmt(a['path_ratio'])} | {fmt(a['min_clearance_mean_m'])} / {fmt(a['min_clearance_worst_m'])} | "
                f"{fmt(a['thrust_impulse_success_ns'], 0)} |"
            )
    note = ("\nsuccess / collision / timeout: fractions of all episodes (timeout includes tilt failures). Time to "
            "goal, path ratio (path length / straight start-goal distance) and thrust impulse: mean over successful "
            "episodes. Min clearance: true gap between the vehicle bounding sphere and the nearest obstacle or the "
            "seabed, per episode; mean and worst over all episodes.\n")
    return "\n".join(lines) + "\n" + note


def load_rl(policy_dir: pathlib.Path, cfg: NavConfig) -> NavController:
    from rov_rl.policy_io import CONFIG_FILE, load_policy  # needs jax + brax

    net_cfg = json.loads((policy_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    expected = nav_core.nav_obs_size(cfg.sonar)
    if net_cfg.get("task") != "nav" or net_cfg.get("observation_size") != expected or net_cfg.get("action_size") != 4:
        raise ValueError(f"{policy_dir} is not a nav policy for this config (need task=nav, observation_size="
                         f"{expected}, action_size=4; got {net_cfg})")
    return NavPolicyController(load_policy(policy_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controllers", default="baseline,rl", help="comma list of: baseline, rl")
    parser.add_argument("--policy-dir", type=pathlib.Path, help="train.py --task nav output <run>/policy (for rl)")
    parser.add_argument("--baseline-params", type=pathlib.Path, default=DEFAULT_VFH_PATH)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per cell")
    parser.add_argument("--densities", default=",".join(DENSITY_ORDER))
    parser.add_argument("--current-levels", default=",".join(str(x) for x in NAV_CURRENT_LEVELS))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-video", action="store_true", help="skip videos and the overhead plot")
    parser.add_argument("--video-cells", default="high_L2,medium_L0", help="cells to film (first episode)")
    parser.add_argument("--plot-cell", default="high_L2", help="cell whose first field is plotted overhead")
    parser.add_argument("--frame-stride", type=int, default=2, help="control steps per video frame")
    args = parser.parse_args()

    controllers = [c for c in args.controllers.split(",") if c]
    if not controllers or not set(controllers) <= set(CONTROLLERS):
        parser.error(f"--controllers must be a comma list of {', '.join(CONTROLLERS)}")
    if "rl" in controllers and args.policy_dir is None:
        parser.error("--policy-dir is required for the rl controller")
    densities = [d for d in args.densities.split(",") if d]
    if not set(densities) <= set(NAV_DENSITY_LEVELS):
        parser.error(f"--densities must be a comma list of {', '.join(DENSITY_ORDER)}")
    levels = [int(x) for x in args.current_levels.split(",") if x]
    cells = [NavCell(d, lvl) for d in densities for lvl in levels]
    video_cells = set() if args.no_video else {c for c in args.video_cells.split(",") if c}
    plot_cell = None if args.no_video else args.plot_cell

    cfg = NavConfig()
    vfh_params = VfhParams.load(args.baseline_params)
    impls: dict[str, NavController] = {}
    if "baseline" in controllers:
        impls["baseline"] = VfhController(VfhPlanner(vfh_params, cfg.sonar, cfg.command))
    if "rl" in controllers:
        impls["rl"] = load_rl(args.policy_dir, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(cells) * len(controllers) * args.episodes
    done = 0
    walls: list[float] = []
    counts = {"success": 0, "collision": 0}
    rows: list[dict[str, Any]] = []
    all_metrics: dict[str, list[dict[str, Any]]] = {c: [] for c in controllers}
    t_run = time.monotonic()
    for cell in cells:
        env = NavCpuEnv(nav_eval_vehicle(cell.level), nav_eval_scene(cell.density), cfg)
        episodes = cell_episodes(env, cell, args.episodes, args.seed)
        row: dict[str, Any] = {"density": cell.density, "level": cell.level, "aggregate": {}, "episodes": {}}
        kept: dict[str, NavEpisodeLog] = {}
        for name in controllers:
            per_ep = []
            for i, (params, noise_seed) in enumerate(episodes):
                t0 = time.monotonic()
                log = run_nav_episode(env, impls[name], params, noise_seed)
                m = nav_episode_metrics(log, params)
                wall = time.monotonic() - t0
                per_ep.append(m)
                done += 1
                walls.append(wall)
                counts["success"] += int(m["outcome"] == "success")
                counts["collision"] += int(m["outcome"] == "collision")
                print(
                    f"[nav_eval] {done:>5d}/{total} ctrl={name} cell={cell.name} ep={i + 1}/{args.episodes} "
                    f"outcome={m['outcome']} t={log.t_s[-1]:.1f}s ratio={fmt(m['path_ratio'])} "
                    f"min_clear={m['min_clearance_m']:.2f}m impulse={m['thrust_impulse_ns']:.0f}Ns "
                    f"obstacles={m['num_obstacles']:.0f} wall={wall:.1f}s",
                    flush=True,
                )
                if done % TALLY_EVERY == 0 or done == total:
                    med = statistics.median(walls)
                    print(
                        f"[nav_eval] tally {done}/{total} success_so_far={counts['success'] / done:.3f} "
                        f"collision_so_far={counts['collision'] / done:.3f} median_wall={med:.1f}s "
                        f"eta={(total - done) * med / 60.0:.1f}min",
                        flush=True,
                    )
                if i == 0 and (cell.name in video_cells or cell.name == plot_cell):
                    kept[name] = log
            row["aggregate"][name] = aggregate_nav(per_ep)
            row["episodes"][name] = per_ep
            all_metrics[name].extend(per_ep)
        rows.append(row)
        if kept:
            write_media(args.out_dir, cell, episodes[0][0], kept, cfg, video_cells, plot_cell, args.frame_stride)
        summary = {"density": "all", "level": -1, "aggregate": {c: aggregate_nav(all_metrics[c]) for c in controllers}}
        meta = {"controllers": controllers, "episodes_per_cell": args.episodes, "seed": args.seed,
                "policy_dir": str(args.policy_dir) if args.policy_dir else None,
                "baseline_params": dataclasses.asdict(vfh_params), "nav_config": dataclasses.asdict(cfg),
                "density_levels": NAV_DENSITY_LEVELS, "disturbance_levels_mps": DISTURBANCE_LEVELS_MPS,
                "elapsed_s": round(time.monotonic() - t_run, 1)}
        (args.out_dir / "metrics.json").write_text(
            json.dumps(json_safe({"meta": meta, "cells": rows, "all": summary}), indent=1) + "\n", encoding="utf-8")
        (args.out_dir / "results.md").write_text(results_table(rows + [summary], controllers), encoding="utf-8")
    print(f"[nav_eval] finished {total} episodes in {(time.monotonic() - t_run) / 60.0:.1f} min", flush=True)
    print(results_table(rows + [summary], controllers), flush=True)


def write_media(out_dir: pathlib.Path, cell: NavCell, params: nav_core.NavEpisode, logs: dict[str, NavEpisodeLog],
                cfg: NavConfig, video_cells: set[str], plot_cell: str | None, frame_stride: int) -> None:
    from rov_rl.nav_render import plot_paths, write_nav_videos

    if cell.name == plot_cell:
        path = out_dir / f"paths_{cell.name}.png"
        plot_paths(params, logs, path, f"nav field {cell.name}: density {cell.density}, current level {cell.level}",
                   cfg.reward.goal_radius_m)
        print(f"[nav_eval] plot -> {path}", flush=True)
    if cell.name in video_cells:
        for path in write_nav_videos(out_dir, f"nav_{cell.name}", params, logs, cfg.sonar, cfg.sim.ctrl_dt_s,
                                     frame_stride, pair=("rl", "baseline")):
            print(f"[nav_eval] video -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
