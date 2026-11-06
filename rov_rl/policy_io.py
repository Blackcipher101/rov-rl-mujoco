"""Save / load trained Brax PPO policies (requires jax + brax)."""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any, Callable

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

PARAMS_FILE = "params.pkl"
CONFIG_FILE = "policy_config.json"


def make_networks(net_cfg: dict[str, Any]) -> ppo_networks.PPONetworks:
    return ppo_networks.make_ppo_networks(
        net_cfg["observation_size"],
        net_cfg["action_size"],
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(net_cfg["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(net_cfg["value_hidden_layer_sizes"]),
    )


def save_policy(out_dir: pathlib.Path, params: Any, net_cfg: dict[str, Any]) -> None:
    """Write params (normalizer, policy, value) as host numpy + the network config.

    Writes to temporary files first so an interrupted save never leaves a
    truncated checkpoint behind.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    host_params = jax.tree_util.tree_map(np.asarray, jax.device_get(params))
    tmp = out_dir / (PARAMS_FILE + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(host_params, f)
    tmp.replace(out_dir / PARAMS_FILE)
    tmp_cfg = out_dir / (CONFIG_FILE + ".tmp")
    tmp_cfg.write_text(json.dumps(net_cfg, indent=2) + "\n", encoding="utf-8")
    tmp_cfg.replace(out_dir / CONFIG_FILE)


def load_policy(policy_dir: pathlib.Path) -> Callable[[np.ndarray], np.ndarray]:
    """Deterministic (distribution mode) jitted policy: obs (OBS_SIZE,) -> action (8,).

    Only load checkpoints you produced yourself: the params file is a pickle.
    """
    net_cfg = json.loads((policy_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    with open(policy_dir / PARAMS_FILE, "rb") as f:
        params = pickle.load(f)
    networks = make_networks(net_cfg)
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    key = jax.random.PRNGKey(0)  # unused by the deterministic policy

    def act(obs: np.ndarray) -> np.ndarray:
        action, _ = policy(np.asarray(obs, dtype=np.float32)[None], key)
        return np.asarray(action[0], dtype=np.float64)

    return act
