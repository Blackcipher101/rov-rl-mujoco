"""Train the ROV controller with MJX + Brax PPO and domain randomization.

Example (see scripts/run_ws1.sh):
    python train.py --preset smoke --out-dir runs/smoke
    python train.py --preset full  --out-dir runs/full
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

    from rov_rl import core
    from rov_rl.config import TRAIN_RANDOMIZATION, RewardConfig, SimConfig
    from rov_rl.mjx_env import RovMjxEnv
    from rov_rl.policy_io import save_policy

    sim = SimConfig()
    env = RovMjxEnv(TRAIN_RANDOMIZATION, sim, RewardConfig())
    ppo_cfg = {**COMMON_PPO, **PRESETS[args.preset], "episode_length": sim.episode_steps, "seed": args.seed}
    if args.num_timesteps is not None:
        ppo_cfg["num_timesteps"] = args.num_timesteps

    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(__file__).resolve().parent
    run_meta = {
        "preset": args.preset,
        "ppo": ppo_cfg,
        "network": NETWORK,
        "sim": dataclasses.asdict(sim),
        "train_randomization": dataclasses.asdict(TRAIN_RANDOMIZATION),
        "git_revision": git_revision(repo),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    print(f"[train] devices={jax.devices()} preset={args.preset} out={args.out_dir}", flush=True)

    net_cfg = {"observation_size": core.OBS_SIZE, "action_size": env.action_size, **NETWORK}
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
        mean_err = float(metrics.get("eval/episode_pos_err_m", float("nan"))) / ep_len
        print(
            f"[train] step={step:>11d} t={elapsed:7.1f}s "
            f"eval_return={float(metrics.get('eval/episode_reward', float('nan'))):9.2f} "
            f"+-{float(metrics.get('eval/episode_reward_std', float('nan'))):7.2f} "
            f"ep_len={ep_len:6.1f} mean_pos_err={mean_err:6.3f} m",
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
