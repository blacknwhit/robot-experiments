"""Serve the pinned OpenPI policy with request-local deterministic diffusion noise."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import jax
import jax.numpy as jnp
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


POLICY_RNG_MODE = "stateless_request_v1"
POLICY_SEED_VERSION = "sha256_u32_v1"
POLICY_RNG_SEED_KEY = "__pi05_rng_seed"


class DeterministicPolicy(_base_policy.BasePolicy):
    """Inject explicit noise so inference is independent of server history."""

    def __init__(self, policy: _base_policy.BasePolicy, *, action_horizon: int, action_dim: int):
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._metadata = {
            **policy.metadata,
            "policy_rng_mode": POLICY_RNG_MODE,
            "policy_seed_version": POLICY_SEED_VERSION,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        request = dict(obs)
        try:
            request_seed = request.pop(POLICY_RNG_SEED_KEY)
        except KeyError as exc:
            raise KeyError(f"Missing required request field {POLICY_RNG_SEED_KEY!r}") from exc
        if isinstance(request_seed, bool) or not isinstance(request_seed, int):
            raise TypeError(f"{POLICY_RNG_SEED_KEY} must be an integer, got {type(request_seed).__name__}")
        if not 0 <= request_seed < 2**32:
            raise ValueError(f"{POLICY_RNG_SEED_KEY} must be in [0, 2**32), got {request_seed}")

        noise = jax.random.normal(
            jax.random.key(request_seed),
            (self._action_horizon, self._action_dim),
            dtype=jnp.float32,
        )
        return self._policy.infer(request, noise=noise)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-config", default="pi05_pretrain_human300")
    parser.add_argument("--policy-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(train_config, args.policy_dir)
    policy = DeterministicPolicy(
        policy,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
    )
    logging.info(
        "Serving deterministic policy RNG mode=%s seed_version=%s",
        POLICY_RNG_MODE,
        POLICY_SEED_VERSION,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
