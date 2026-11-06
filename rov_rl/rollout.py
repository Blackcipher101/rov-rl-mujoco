"""Episode rollout on the CPU env and per-episode metrics."""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Protocol

import numpy as np

from rov_rl import core
from rov_rl.config import MetricConfig
from rov_rl.cpu_env import RovCpuEnv
from rov_rl.pid_baseline import CascadePid


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, env: RovCpuEnv, obs: np.ndarray) -> np.ndarray: ...


class PidController:
    def __init__(self, pid: CascadePid) -> None:
        self.pid = pid

    def reset(self) -> None:
        self.pid.reset()

    def act(self, env: RovCpuEnv, obs: np.ndarray) -> np.ndarray:
        return self.pid.act(env.data.qpos, env.data.qvel, env.reference())


class PolicyController:
    """Wraps any obs -> action function (e.g. a jitted Brax PPO policy)."""

    def __init__(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        self.policy = policy

    def reset(self) -> None:
        return None

    def act(self, env: RovCpuEnv, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy(obs), dtype=np.float64)


@dataclasses.dataclass
class EpisodeLog:
    is_station: bool
    path_end_time_s: float
    t_s: np.ndarray
    pos_err_m: np.ndarray
    yaw_err_rad: np.ndarray
    thrust_impulse_ns: np.ndarray
    qpos: np.ndarray  # (steps + 1, 7), includes the initial state
    ref_pos_m: np.ndarray  # (steps + 1, 3)
    terminated: bool
    episode_return: float


def run_episode(env: RovCpuEnv, controller: Controller, params: core.EpisodeParams) -> EpisodeLog:
    obs = env.reset(params=params)
    controller.reset()
    t, pe, ye, imp = [], [], [], []
    qpos = [env.data.qpos.copy()]
    ref_pos = [env.reference().pos_m.copy()]
    ret = 0.0
    terminated = False
    while True:
        obs, reward, done, info = env.step(controller.act(env, obs))
        ret += reward
        t.append(info.t_s)
        pe.append(info.pos_err_m)
        ye.append(info.yaw_err_rad)
        imp.append(info.thrust_impulse_ns)
        qpos.append(env.data.qpos.copy())
        ref_pos.append(env.reference().pos_m.copy())
        if done:
            terminated = info.terminated
            break
    return EpisodeLog(
        is_station=bool(params.is_station > 0.5),
        path_end_time_s=float(core.path_end_time_s(np, params)),
        t_s=np.array(t),
        pos_err_m=np.array(pe),
        yaw_err_rad=np.array(ye),
        thrust_impulse_ns=np.array(imp),
        qpos=np.array(qpos),
        ref_pos_m=np.array(ref_pos),
        terminated=terminated,
        episode_return=ret,
    )


def settling_time_s(t_s: np.ndarray, pos_err_m: np.ndarray, yaw_err_rad: np.ndarray, cfg: MetricConfig) -> float:
    """First time after which position and yaw errors stay inside the band; NaN if never."""
    inside = (pos_err_m < cfg.settle_pos_m) & (np.abs(yaw_err_rad) < cfg.settle_yaw_rad)
    if inside.size == 0 or not inside[-1]:
        return math.nan
    outside_idx = np.flatnonzero(~inside)
    if outside_idx.size == 0:
        return 0.0
    return float(t_s[outside_idx[-1]])  # last sample outside the band ends the transient


def episode_metrics(log: EpisodeLog, cfg: MetricConfig, episode_duration_s: float) -> dict[str, float | bool]:
    """Metrics for one episode. Error statistics are NaN for terminated episodes."""
    if log.is_station:
        window = log.t_s >= cfg.station_window_start_s
    else:
        window = np.ones_like(log.t_s, dtype=bool)
    completed = (not log.terminated) and log.t_s.size > 0 and log.t_s[-1] >= episode_duration_s - 1e-6
    if completed and np.any(window):
        e = log.pos_err_m[window]
        rms = float(np.sqrt(np.mean(e**2)))
        peak = float(np.max(e))
        yaw_rms = float(np.sqrt(np.mean(log.yaw_err_rad[window] ** 2)))
    else:
        rms = peak = yaw_rms = math.nan
    settle = settling_time_s(log.t_s, log.pos_err_m, log.yaw_err_rad, cfg) if completed else math.nan
    if not completed:
        success = False
    elif log.is_station:
        success = peak <= cfg.station_success_pos_m
    else:
        success = rms <= cfg.track_success_rms_m and float(log.pos_err_m[-1]) <= cfg.track_success_final_m
    return {
        "pos_rms_m": rms,
        "pos_max_m": peak,
        "yaw_rms_rad": yaw_rms,
        "settling_time_s": settle,
        "thrust_impulse_ns": float(np.sum(log.thrust_impulse_ns)),
        "terminated": bool(log.terminated),
        "success": bool(success),
        "episode_return": float(log.episode_return),
    }


def aggregate(metrics: list[dict[str, float | bool]]) -> dict[str, float]:
    """Mean over episodes (NaN-aware for error metrics) plus rates."""
    out: dict[str, float] = {"episodes": float(len(metrics))}
    if not metrics:
        return out
    for key in ("pos_rms_m", "pos_max_m", "yaw_rms_rad", "settling_time_s", "thrust_impulse_ns", "episode_return"):
        vals = np.array([float(m[key]) for m in metrics])
        finite = vals[np.isfinite(vals)]
        out[key] = float(np.mean(finite)) if finite.size else math.nan
    out["success_rate"] = float(np.mean([bool(m["success"]) for m in metrics]))
    out["termination_rate"] = float(np.mean([bool(m["terminated"]) for m in metrics]))
    out["settled_rate"] = float(np.mean([math.isfinite(float(m["settling_time_s"])) for m in metrics]))
    return out
