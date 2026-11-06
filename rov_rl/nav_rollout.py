"""Nav-task episode runner on the CPU env, per-episode metrics and aggregation."""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Protocol

import numpy as np

from rov_rl import nav_core
from rov_rl.nav_baseline import VfhPlanner
from rov_rl.nav_cpu_env import NavCpuEnv

OUTCOMES = ("success", "collision", "timeout", "failure")


class NavController(Protocol):
    def reset(self) -> None: ...

    def act(self, meas: nav_core.NavMeasurement, obs: np.ndarray) -> np.ndarray: ...


class VfhController:
    def __init__(self, planner: VfhPlanner) -> None:
        self.planner = planner

    def reset(self) -> None:
        self.planner.reset()

    def act(self, meas: nav_core.NavMeasurement, obs: np.ndarray) -> np.ndarray:
        return self.planner.act(meas)


class NavPolicyController:
    """Wraps an obs -> normalized velocity command function (e.g. a jitted Brax PPO policy)."""

    def __init__(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        self.policy = policy

    def reset(self) -> None:
        return None

    def act(self, meas: nav_core.NavMeasurement, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy(obs), dtype=np.float64)


@dataclasses.dataclass
class NavEpisodeLog:
    t_s: np.ndarray  # (steps,)
    qpos: np.ndarray  # (steps + 1, 7), includes the initial state
    sonar_range_m: np.ndarray  # (steps + 1, num_beams) measured fan ranges (what the controller saw)
    clearance_m: np.ndarray  # (steps + 1,) true geometric clearance
    thrust_impulse_ns: np.ndarray  # (steps,)
    cmd: np.ndarray  # (steps, 4)
    outcome: str  # one of OUTCOMES
    episode_return: float


def run_nav_episode(env: NavCpuEnv, controller: NavController, params: nav_core.NavEpisode,
                    noise_seed: int) -> NavEpisodeLog:
    obs = env.reset(params=params, noise_seed=noise_seed)
    controller.reset()
    t, imp, cmds = [], [], []
    qpos = [env.data.qpos.copy()]
    sonar = [np.asarray(env.measurement.sonar_range_m).copy()]
    clearance = [env.clearance_m()]
    ret = 0.0
    while True:
        cmd = controller.act(env.measurement, obs)
        obs, reward, done, info = env.step(cmd)
        ret += reward
        t.append(info.t_s)
        imp.append(info.thrust_impulse_ns)
        cmds.append(np.clip(cmd, -1.0, 1.0))
        qpos.append(env.data.qpos.copy())
        sonar.append(np.asarray(env.measurement.sonar_range_m).copy())
        clearance.append(info.clearance_m)
        if done:
            if info.success:
                outcome = "success"
            elif info.collision:
                outcome = "collision"
            elif info.failure:
                outcome = "failure"
            else:
                outcome = "timeout"
            break
    return NavEpisodeLog(
        t_s=np.array(t),
        qpos=np.array(qpos),
        sonar_range_m=np.array(sonar),
        clearance_m=np.array(clearance),
        thrust_impulse_ns=np.array(imp),
        cmd=np.array(cmds),
        outcome=outcome,
        episode_return=ret,
    )


def nav_episode_metrics(log: NavEpisodeLog, params: nav_core.NavEpisode) -> dict[str, float | str | bool]:
    """Per-episode metrics. Time to goal and path ratio are NaN unless the episode succeeded.

    path_ratio = travelled 3D path length / straight start-goal distance (>= ~1; the goal radius lets a
    direct run come in slightly below 1).
    """
    steps = np.diff(log.qpos[:, :3], axis=0)
    path = float(np.sum(np.linalg.norm(steps, axis=-1)))
    straight = float(np.linalg.norm(np.asarray(params.scene.goal_m) - log.qpos[0, :3]))
    success = log.outcome == "success"
    return {
        "outcome": log.outcome,
        "success": success,
        "time_to_goal_s": float(log.t_s[-1]) if success else math.nan,
        "path_length_m": path,
        "straight_line_m": straight,
        "path_ratio": path / straight if success and straight > 0.0 else math.nan,
        "min_clearance_m": float(np.min(log.clearance_m)),
        "thrust_impulse_ns": float(np.sum(log.thrust_impulse_ns)),
        "episode_return": float(log.episode_return),
        "num_obstacles": float(np.sum(params.scene.cap_active) + np.sum(params.scene.box_active)),
    }


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else math.nan


def aggregate_nav(metrics: list[dict[str, float | str | bool]]) -> dict[str, float]:
    """Outcome rates over all episodes; time, path ratio and impulse over successful episodes only
    (comparable across controllers); minimum clearance over all episodes."""
    out: dict[str, float] = {"episodes": float(len(metrics))}
    if not metrics:
        return out
    for o in OUTCOMES:
        out[f"{o}_rate"] = float(np.mean([m["outcome"] == o for m in metrics]))
    ok = [m for m in metrics if m["success"]]
    out["time_to_goal_s"] = _nanmean([float(m["time_to_goal_s"]) for m in ok])
    out["path_ratio"] = _nanmean([float(m["path_ratio"]) for m in ok])
    out["thrust_impulse_success_ns"] = _nanmean([float(m["thrust_impulse_ns"]) for m in ok])
    out["min_clearance_mean_m"] = _nanmean([float(m["min_clearance_m"]) for m in metrics])
    out["min_clearance_worst_m"] = float(np.min([float(m["min_clearance_m"]) for m in metrics]))
    out["episode_return"] = _nanmean([float(m["episode_return"]) for m in metrics])
    return out
