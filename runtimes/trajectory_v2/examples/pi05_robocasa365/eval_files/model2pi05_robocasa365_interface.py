"""OpenPI PI0.5 policy adapter for StarVLA's RoboCasa365 simulator harness.

The OpenPI websocket policy accepts one unbatched observation per request. For
vectorized simulator observations this adapter deliberately sends one request
per environment and stacks the returned chunks afterwards.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


VIDEO_KEYS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}

STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)

# Official OpenPI RoboCasa output order. This is intentionally different from
# the canonical numeric order used by the existing StarVLA checkpoint adapter.
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}

POLICY_RNG_MODE = "stateless_request_v1"
POLICY_SEED_VERSION = "sha256_u32_v1"
POLICY_RNG_SEED_KEY = "__pi05_rng_seed"
STRICT_DETERMINISM_VERSION = "pytorch_eager_cublas_v1"


def _hash_value(hasher: Any, value: Any) -> None:
    """Hash nested policy inputs/outputs without relying on repr or hash()."""
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value):
            _hash_value(hasher, str(key))
            _hash_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"list" if isinstance(value, list) else b"tuple")
        hasher.update(struct.pack(">Q", len(value)))
        for item in value:
            _hash_value(hasher, item)
        return
    if isinstance(value, str):
        data = value.encode("utf-8")
        hasher.update(b"str")
        hasher.update(struct.pack(">Q", len(data)))
        hasher.update(data)
        return
    if value is None:
        hasher.update(b"none")
        return
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError(f"Cannot deterministically hash object dtype {array.dtype}")
    hasher.update(b"array")
    _hash_value(hasher, array.dtype.str)
    hasher.update(struct.pack(">Q", array.ndim))
    for size in array.shape:
        hasher.update(struct.pack(">Q", int(size)))
    hasher.update(array.tobytes(order="C"))


def stable_request_seed(
    base_seed: int,
    task_name: str,
    environment_index: int,
    episode_index: int,
    replan_index: int,
) -> int:
    """Derive a stable uint32 seed without depending on Python's hash salt."""
    fields = (
        POLICY_SEED_VERSION,
        str(int(base_seed)),
        task_name,
        str(int(environment_index)),
        str(int(episode_index)),
        str(int(replan_index)),
    )
    digest = hashlib.sha256("\0".join(fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match the official openpi-client PIL resize-with-zero-padding helper."""
    image = np.asarray(image)
    if image.shape[-3:-1] == (height, width):
        return image
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    pil_image = Image.fromarray(image)
    cur_width, cur_height = pil_image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded.paste(resized, (pad_width, pad_height))
    assert padded.size == (width, height)
    return np.asarray(padded)


def _batch_size(observations: Dict[str, Any]) -> int:
    value = np.asarray(observations[VIDEO_KEYS["observation/image"]])
    if value.ndim != 5:
        raise ValueError(f"Expected batched video shape [B,T,H,W,C], got {value.shape}")
    return int(value.shape[0])


def _latest(observations: Dict[str, Any], key: str, batch_index: int) -> np.ndarray:
    value = np.asarray(observations[key])
    if value.ndim < 3:
        raise ValueError(f"Expected batched temporal value for {key}, got {value.shape}")
    return value[batch_index, -1]


def _instruction_at(value: Any, batch_index: int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return str(value[batch_index])
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    return str(array[batch_index])


class PolicyWrapper:
    """Convert StarVLA RoboCasa365 observations into official OpenPI inputs."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        image_size: tuple[int, int] = (224, 224),
        n_action_steps: int = 5,
        task_name: str = "",
        base_seed: int = 7,
        strict_determinism: bool = False,
        trace_determinism: bool = False,
        determinism_version: str = STRICT_DETERMINISM_VERSION,
        client: Optional[Any] = None,
    ) -> None:
        self.client = client if client is not None else WebsocketClientPolicy(host, port)
        self.image_size = tuple(image_size)
        self.n_action_steps = int(n_action_steps)
        self.task_name = str(task_name)
        self.base_seed = int(base_seed)
        self.strict_determinism = bool(strict_determinism)
        self.trace_determinism = bool(trace_determinism)
        self.determinism_version = str(determinism_version)
        self._episode_indices: list[int] = []
        self._replan_indices: list[int] = []
        self._trace_hashers: list[Any] = []
        self._episode_trace_hashes: list[str] = []
        metadata = self.client.get_server_metadata()
        self.server_metadata = dict(metadata)
        self._native_starvla = metadata.get("action_output_space") == "environment"
        server_rng_mode = metadata.get("policy_rng_mode")
        server_seed_version = metadata.get("policy_seed_version")
        if server_rng_mode != POLICY_RNG_MODE or server_seed_version != POLICY_SEED_VERSION:
            raise RuntimeError(
                "OpenPI server does not advertise the required deterministic RNG contract: "
                f"expected ({POLICY_RNG_MODE!r}, {POLICY_SEED_VERSION!r}), "
                f"got ({server_rng_mode!r}, {server_seed_version!r})"
            )
        if self.strict_determinism:
            if metadata.get("strict_determinism") is not True:
                raise RuntimeError("Policy server does not advertise strict deterministic execution")
            if metadata.get("determinism_version") != self.determinism_version:
                raise RuntimeError(
                    "Policy server determinism contract mismatch: "
                    f"expected {self.determinism_version!r}, "
                    f"got {metadata.get('determinism_version')!r}"
                )
        print(f"[pi05] connected to OpenPI server: {metadata}")

    @property
    def episode_trace_hashes(self) -> list[str]:
        return list(self._episode_trace_hashes)

    def _new_trace_hasher(self, batch_index: int) -> Any:
        hasher = hashlib.sha256()
        _hash_value(
            hasher,
            {
                "trace_version": "pi05_observation_action_sha256_v1",
                "determinism_version": self.determinism_version,
                "task_name": self.task_name,
                "base_seed": self.base_seed,
                "environment_index": batch_index,
                "episode_index": self._episode_indices[batch_index],
            },
        )
        return hasher

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _make_openpi_observation(self, observations: Dict[str, Any], batch_index: int) -> dict:
        height, width = self.image_size
        element = {
            openpi_key: convert_to_uint8(
                resize_with_pad(_latest(observations, source_key, batch_index), height, width)
            )
            for openpi_key, source_key in VIDEO_KEYS.items()
        }
        state_parts = [np.asarray(_latest(observations, key, batch_index)) for key in STATE_KEYS]
        state = np.concatenate(state_parts, axis=-1)
        if state.shape != (16,):
            raise ValueError(f"Expected a 16-D RoboCasa state, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("RoboCasa state contains NaN or Inf")
        element["observation/state"] = state
        element["prompt"] = _instruction_at(
            observations["annotation.human.task_description"], batch_index
        )
        return element

    def _make_starvla_example(self, observations: Dict[str, Any], batch_index: int) -> dict:
        images = [
            convert_to_uint8(_latest(observations, VIDEO_KEYS[key], batch_index))
            for key in ("observation/image", "observation/wrist_image", "observation/right_image")
        ]
        state = np.concatenate(
            [np.asarray(_latest(observations, key, batch_index)) for key in STATE_KEYS],
            axis=-1,
        ).astype(np.float32, copy=False)
        if state.shape != (16,) or not np.all(np.isfinite(state)):
            raise ValueError(f"Expected finite 16-D RoboCasa state, got {state.shape}")
        return {
            "image": images,
            "state": state,
            "lang": _instruction_at(observations["annotation.human.task_description"], batch_index),
            "robot_tag": "pi05_panda_omron_robocasa365",
        }

    def _ensure_counters(self, batch_size: int) -> None:
        if not self._episode_indices:
            self._episode_indices = [0] * batch_size
            self._replan_indices = [0] * batch_size
            if self.trace_determinism:
                self._trace_hashers = [self._new_trace_hasher(i) for i in range(batch_size)]
        elif len(self._episode_indices) != batch_size:
            raise ValueError(
                "RoboCasa vector batch size changed during evaluation: "
                f"{len(self._episode_indices)} -> {batch_size}"
            )

    def finish_step(self, done: Any) -> None:
        """Advance episode-local counters after the simulator consumes a chunk."""
        done_array = np.asarray(done, dtype=bool).reshape(-1)
        self._ensure_counters(len(done_array))
        for batch_index, is_done in enumerate(done_array):
            if is_done:
                if self.trace_determinism:
                    self._episode_trace_hashes.append(self._trace_hashers[batch_index].hexdigest())
                self._episode_indices[batch_index] += 1
                self._replan_indices[batch_index] = 0
                if self.trace_determinism:
                    self._trace_hashers[batch_index] = self._new_trace_hasher(batch_index)

    def step(self, observations: Dict[str, Any], **_: Any) -> Dict[str, Dict[str, np.ndarray]]:
        batch_size = _batch_size(observations)
        self._ensure_counters(batch_size)
        chunks = []
        for batch_index in range(batch_size):
            element = (
                self._make_starvla_example(observations, batch_index)
                if self._native_starvla
                else self._make_openpi_observation(observations, batch_index)
            )
            element[POLICY_RNG_SEED_KEY] = stable_request_seed(
                self.base_seed,
                self.task_name,
                batch_index,
                self._episode_indices[batch_index],
                self._replan_indices[batch_index],
            )
            if self.trace_determinism:
                _hash_value(self._trace_hashers[batch_index], {"request": element})
            response = self.client.predict_action({"examples": [element]}) if self._native_starvla else self.client.predict_action(element)
            self._replan_indices[batch_index] += 1
            if self._native_starvla:
                if response.get("status") != "ok":
                    raise RuntimeError(f"StarVLA policy server error: {response}")
                response = response["data"]
            if "actions" not in response:
                raise KeyError(f"OpenPI response has no 'actions': {response.keys()}")
            actions = np.asarray(response["actions"])
            if self._native_starvla:
                if actions.ndim != 3 or actions.shape[0] != 1:
                    raise ValueError(f"Expected native StarVLA actions [1,T,D], got {actions.shape}")
                actions = actions[0]
            if actions.ndim != 2 or actions.shape[0] < self.n_action_steps or actions.shape[1] < 12:
                raise ValueError(
                    f"Expected OpenPI actions [T>= {self.n_action_steps}, D>=12], got {actions.shape}"
                )
            actions = actions[: self.n_action_steps, :12]
            if not np.all(np.isfinite(actions)):
                raise ValueError("OpenPI actions contain NaN or Inf")
            if self.trace_determinism:
                _hash_value(self._trace_hashers[batch_index], {"full_action_chunk": actions})
            chunks.append(actions)

        raw_actions = np.stack(chunks, axis=0)
        output = {
            key: raw_actions[:, :, start:end]
            for key, (start, end) in ACTION_SLICES.items()
        }
        return {"actions": output}
