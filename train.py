"""Train the ROV controller with MJX + Brax PPO and domain randomization.

Tasks: "control" (default; station keeping + trajectory tracking, thruster-level actions) and "nav" (local
planning / obstacle avoidance with a simulated sonar, velocity-command actions; rov_rl/nav_mjx_env.py).

Example (see scripts/run_ws1.sh):
    python train.py --preset smoke --out-dir runs/smoke
    python train.py --preset full  --out-dir runs/full
    python train.py --task nav --preset smoke --out-dir runs/nav_smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    # ~61k env steps per PPO iteration; checks the pipeline end to end.
    "smoke": dict(
        num_timesteps=1_000_000,
        num_evals=3,
        num_envs=256,
        batch_size=256,
        num_minibatches=8,
        num_eval_envs=64,
    ),
    "full": dict(
        num_timesteps=150_000_000,
        num_evals=25,
        num_envs=2048,
        batch_size=1024,
        num_minibatches=32,
        num_eval_envs=256,
    ),
}

# Nav episodes last up to 100 s (2500 control steps); obstacle field and current are re-sampled every episode.
NAV_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(
        num_timesteps=2_000_000,
        num_evals=3,
        num_envs=256,
        batch_size=256,
        num_minibatches=8,
        num_eval_envs=64,
    ),
    "full": dict(
        num_timesteps=300_000_000,
        num_evals=25,
        num_envs=2048,
        batch_size=1024,
        num_minibatches=32,
        num_eval_envs=256,
    ),
}
# Longer discount horizon for nav (1 / (1 - 0.995) = 200 steps = 8 s, about 4 m at cruise speed), so the value
# of avoiding an obstacle a few metres ahead propagates back to the steering decision.
NAV_PPO_OVERRIDES = dict(discounting=0.995)

COMMON_PPO = dict(
    episode_length=None,  # filled from SimConfig
    action_repeat=1,
    unroll_length=30,
    num_updates_per_batch=8,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=5e-3,
    reward_scaling=1.0,
    normalize_observations=True,
    clipping_epsilon=0.2,
    max_grad_norm=1.0,
)

NETWORK = dict(policy_hidden_layer_sizes=[128, 128, 128, 128], value_hidden_layer_sizes=[256, 256, 256, 256, 256])


def git_revision(repo: pathlib.Path) -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("control", "nav"), default="control")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-timesteps", type=int, help="override the preset")
    args = parser.parse_args()

    # Imported here so `--help` works without a GPU stack.
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground._src import wrapper

    from rov_rl import core, nav_core
    from rov_rl.config import (NAV_TRAIN_SCENE, NAV_TRAIN_VEHICLE, TRAIN_RANDOMIZATION, NavConfig, RewardConfig,
                               SimConfig)
    from rov_rl.policy_io import save_policy

    if args.task == "control":
        from rov_rl.mjx_env import RovMjxEnv

        sim = SimConfig()
        env = RovMjxEnv(TRAIN_RANDOMIZATION, sim, RewardConfig())
        presets, overrides, obs_size = PRESETS, {}, core.OBS_SIZE
        task_meta: dict[str, Any] = {"train_randomization": dataclasses.asdict(TRAIN_RANDOMIZATION)}
    else:
        from rov_rl.nav_mjx_env import NavMjxEnv

        nav_cfg = NavConfig()
        sim = nav_cfg.sim
        env = NavMjxEnv(NAV_TRAIN_VEHICLE, NAV_TRAIN_SCENE, nav_cfg)
        presets, overrides, obs_size = NAV_PRESETS, NAV_PPO_OVERRIDES, nav_core.nav_obs_size(nav_cfg.sonar)
        task_meta = {"train_vehicle": dataclasses.asdict(NAV_TRAIN_VEHICLE),
                     "train_scene": dataclasses.asdict(NAV_TRAIN_SCENE), "nav_config": dataclasses.asdict(nav_cfg)}
    ppo_cfg = {**COMMON_PPO, **overrides, **presets[args.preset], "episode_length": sim.episode_steps,
               "seed": args.seed}
    if args.num_timesteps is not None:
        ppo_cfg["num_timesteps"] = args.num_timesteps

    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(__file__).resolve().parent
    run_meta = {
        "task": args.task,
        "preset": args.preset,
        "ppo": ppo_cfg,
        "network": NETWORK,
        "sim": dataclasses.asdict(sim),
        **task_meta,
        "git_revision": git_revision(repo),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    print(f"[train] devices={jax.devices()} task={args.task} preset={args.preset} out={args.out_dir}", flush=True)

    net_cfg = {"task": args.task, "observation_size": obs_size, "action_size": env.action_size, **NETWORK}
    progress_path = args.out_dir / "progress.jsonl"
    t_start = time.monotonic()
    times = {"first_progress_s": None}

    def progress(step: int, metrics: dict[str, Any]) -> None:
        elapsed = time.monotonic() - t_start
        if times["first_progress_s"] is None:
            times["first_progress_s"] = elapsed  # includes JIT compilation
        row = {"step": int(step), "elapsed_s": round(elapsed, 1)}
        row.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/") or k.startswith("training/")})
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ep_len = max(float(metrics.get("eval/avg_episode_length", sim.episode_steps)), 1.0)
        if args.task == "control":
            mean_err = float(metrics.get("eval/episode_pos_err_m", float("nan"))) / ep_len
            task_str = f"mean_pos_err={mean_err:6.3f} m"
        else:  # per-episode sums of 0/1 flags = rates over the eval episodes
            task_str = (f"success={float(metrics.get('eval/episode_success', float('nan'))):5.3f} "
                        f"collision={float(metrics.get('eval/episode_collision', float('nan'))):5.3f}")
        print(
            f"[train] step={step:>11d} t={elapsed:7.1f}s "
            f"eval_return={float(metrics.get('eval/episode_reward', float('nan'))):9.2f} "
            f"+-{float(metrics.get('eval/episode_reward_std', float('nan'))):7.2f} "
            f"ep_len={ep_len:6.1f} {task_str}",
            flush=True,
        )

    def save_latest(step: int, make_policy: Any, params: Any) -> None:
        save_policy(args.out_dir / "policy", params, {**net_cfg, "step": int(step)})

    network_factory = functools.partial(ppo_networks.make_ppo_networks, **{k: tuple(v) for k, v in NETWORK.items()})
    _, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=save_latest,
        **ppo_cfg,
    )
    save_policy(args.out_dir / "policy", params, {**net_cfg, "step": int(ppo_cfg["num_timesteps"])})
    total = time.monotonic() - t_start
    print(f"[train] done in {total:.1f}s (first progress incl. JIT at {times['first_progress_s']}s); "
          f"policy -> {args.out_dir / 'policy'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
