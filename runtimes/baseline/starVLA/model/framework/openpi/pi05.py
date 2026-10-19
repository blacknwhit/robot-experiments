"""Native StarVLA wrapper around the official OpenPI PyTorch PI0.5 model.

StarVLA owns dataset IO, batching, training, checkpoints, deployment and the
benchmark harness. OpenPI remains the source of truth for the PI0.5 network,
tokenizer, image preprocessing, normalization, loss and action sampling.
"""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, List

import numpy as np
import torch
from transformers import PretrainedConfig

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY


POLICY_RNG_SEED_KEY = "__pi05_rng_seed"

_OFFICIAL_FP32_PARAM_SELECTORS = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)
_OFFICIAL_FP32_TOP_LEVEL_PREFIXES = (
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _configure_gradient_checkpointing(policy_model: Any, enabled: bool) -> None:
    """Make the StarVLA setting authoritative for every OpenPI submodule."""
    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    method = getattr(policy_model, method_name, None)
    if not callable(method):
        raise AttributeError(f"PI0Pytorch does not implement {method_name}()")
    method()


def _tree_map(fn, value):
    if isinstance(value, dict):
        return {key: _tree_map(fn, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [_tree_map(fn, item) for item in value]
    return fn(value)


def _stack_tree(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _stack_tree([item[key] for item in items]) for key in first}
    return np.stack([np.asarray(item) for item in items], axis=0)


@FRAMEWORK_REGISTRY.register("OpenPIPi05")
class OpenPIPi05(baseframework):
    """Thin StarVLA framework adapter for official ``PI0Pytorch``."""

    def __init__(self, config: Any = None, **_: Any) -> None:
        super().__init__(PretrainedConfig())
        pi_cfg = _cfg_get(_cfg_get(config, "framework"), "pi05")
        if pi_cfg is None:
            raise ValueError("OpenPIPi05 requires framework.pi05 configuration")

        self.openpi_root = Path(os.path.expanduser(str(_cfg_get(pi_cfg, "openpi_root")))).resolve()
        openpi_src = self.openpi_root / "src"
        if not openpi_src.is_dir():
            raise FileNotFoundError(f"OpenPI source directory not found: {openpi_src}")
        if str(openpi_src) not in sys.path:
            sys.path.insert(0, str(openpi_src))

        self.config_name = str(_cfg_get(pi_cfg, "openpi_config", "pi05_pretrain_human300"))
        self.assets_dir = Path(os.path.expanduser(str(_cfg_get(pi_cfg, "assets_dir")))).resolve()
        self.num_inference_steps = int(_cfg_get(pi_cfg, "num_inference_steps", 10))

        config_module = importlib.import_module("openpi.training.config")
        pi0_module = importlib.import_module("openpi.models_pytorch.pi0_pytorch")
        normalize_module = importlib.import_module("openpi.shared.normalize")
        transforms_module = importlib.import_module("openpi.transforms")
        model_module = importlib.import_module("openpi.models.model")

        self._model_module = model_module
        self._transforms_module = transforms_module
        self._normalize_module = normalize_module
        self._train_config = config_module.get_config(self.config_name)
        self._model_config = self._train_config.model
        if not bool(getattr(self._model_config, "pi05", False)):
            raise ValueError(f"OpenPI config {self.config_name!r} is not a PI0.5 config")
        action_cfg = _cfg_get(_cfg_get(config, "framework"), "action_model")
        self.action_horizon = int(self._model_config.action_horizon)
        self.environment_action_dim = int(_cfg_get(action_cfg, "action_dim"))
        self.environment_state_dim = int(_cfg_get(action_cfg, "state_dim"))
        self.image_observation_keys = tuple(
            _cfg_get(
                pi_cfg,
                "image_observation_keys",
                ("observation/image", "observation/wrist_image", "observation/right_image"),
            )
        )
        configured_horizon = _cfg_get(action_cfg, "action_horizon")
        if configured_horizon is not None and int(configured_horizon) != self.action_horizon:
            raise ValueError(
                f"StarVLA action_horizon={configured_horizon} does not match OpenPI {self.action_horizon}"
            )

        self.policy_model = pi0_module.PI0Pytorch(self._model_config)
        self.official_precision_mode = bool(
            _cfg_get(pi_cfg, "official_precision_mode", False)
        )
        self.gradient_checkpointing_enabled = bool(_cfg_get(pi_cfg, "gradient_checkpointing", True))
        _configure_gradient_checkpointing(self.policy_model, self.gradient_checkpointing_enabled)

        # Ask the official OpenPI DataConfig for the exact input/model/output
        # transform groups, while loading normalization statistics explicitly
        # from the selected checkpoint assets.
        data_config = self._train_config.data.create(self._train_config.assets_dirs, self._model_config)
        self._use_quantile_norm = bool(data_config.use_quantile_norm)
        self._norm_stats = normalize_module.load(self.assets_dir)
        self._input_transform = transforms_module.compose(
            [
                *data_config.data_transforms.inputs,
                transforms_module.Normalize(self._norm_stats, use_quantiles=self._use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms_module.compose(
            [
                *data_config.model_transforms.outputs,
                transforms_module.Unnormalize(self._norm_stats, use_quantiles=self._use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

        init_checkpoint = _cfg_get(pi_cfg, "init_checkpoint")
        if init_checkpoint:
            self.load_openpi_weights(init_checkpoint)

    @property
    def policy_metadata(self) -> dict[str, Any]:
        return {
            "action_chunk_size": self.action_horizon,
            "action_dim": self.environment_action_dim,
            "state_dim": self.environment_state_dim,
            "image_observation_keys": self.image_observation_keys,
            "action_output_space": "environment",
            "normalization_backend": "openpi",
            "openpi_config": self.config_name,
            "policy_rng_mode": "stateless_request_v1",
            "policy_seed_version": "sha256_u32_v1",
            "training_precision_mode": (
                "openpi_official_selective_bf16"
                if self.official_precision_mode
                else "starvla_legacy_bf16"
            ),
        }

    @staticmethod
    def _official_parameter_dtype(name: str) -> torch.dtype:
        if name.startswith(_OFFICIAL_FP32_TOP_LEVEL_PREFIXES):
            return torch.float32
        if any(selector in name for selector in _OFFICIAL_FP32_PARAM_SELECTORS):
            return torch.float32
        return torch.bfloat16

    def validate_training_precision(self) -> dict[str, int | str | bool]:
        """Assert the post-Accelerate model still has OpenPI's native dtypes."""
        if not self.official_precision_mode:
            return {"enabled": False, "mode": "starvla_legacy_bf16"}

        counts = {"float32_numel": 0, "bfloat16_numel": 0}
        mismatches: list[str] = []
        for name, parameter in self.policy_model.named_parameters():
            if not parameter.is_floating_point():
                continue
            expected = self._official_parameter_dtype(name)
            if parameter.dtype != expected:
                mismatches.append(f"{name}: expected={expected}, actual={parameter.dtype}")
            count_key = "float32_numel" if expected == torch.float32 else "bfloat16_numel"
            counts[count_key] += parameter.numel()

        if mismatches:
            preview = "\n".join(mismatches[:20])
            raise RuntimeError(
                "PI0.5 official precision validation failed after distributed prepare; "
                "the model was likely cast wholesale by the launcher.\n"
                f"{preview}\n"
                f"total_mismatches={len(mismatches)}"
            )
        return {
            "enabled": True,
            "mode": "openpi_official_selective_bf16",
            **counts,
        }

    def _sample_to_openpi(self, sample: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
        images = sample.get("image")
        if not isinstance(images, (list, tuple)) or len(images) != len(self.image_observation_keys):
            raise ValueError(
                f"PI0.5 expects {len(self.image_observation_keys)} images for "
                f"{self.image_observation_keys}, got {type(images)} / {len(images) if images is not None else 0}"
            )
        state = np.asarray(sample["state"], dtype=np.float32).reshape(-1)
        if state.shape != (self.environment_state_dim,):
            raise ValueError(
                f"PI0.5 environment state must be {self.environment_state_dim}-D, got {state.shape}"
            )
        raw = {
            "observation/state": state,
            "prompt": str(sample["lang"]),
        }
        raw.update(
            {
                observation_key: np.asarray(image)
                for observation_key, image in zip(self.image_observation_keys, images, strict=True)
            }
        )
        if include_actions:
            actions = np.asarray(sample["action"], dtype=np.float32)
            if actions.shape != (self.action_horizon, self.environment_action_dim):
                raise ValueError(
                    f"PI0.5 environment actions must be "
                    f"[{self.action_horizon},{self.environment_action_dim}], got {actions.shape}"
                )
            raw["actions"] = actions
        return raw

    def _prepare_batch(self, examples: List[dict], *, include_actions: bool):
        transformed = [
            self._input_transform(self._sample_to_openpi(copy.deepcopy(sample), include_actions=include_actions))
            for sample in examples
        ]
        batch = _stack_tree(transformed)
        device = next(self.policy_model.parameters()).device

        def to_device(value):
            tensor = torch.as_tensor(value)
            return tensor.to(device=device)

        batch = _tree_map(to_device, batch)
        actions = batch.pop("actions", None)
        observation = self._model_module.Observation.from_dict(batch)
        return observation, actions, batch

    def forward(self, examples: List[dict], **kwargs: Any) -> dict[str, torch.Tensor]:
        observation, actions, _ = self._prepare_batch(examples, include_actions=True)
        if actions is None:
            raise ValueError("PI0.5 training batch is missing actions")
        if self.official_precision_mode and torch.is_autocast_enabled():
            raise RuntimeError(
                "PI0.5 official precision mode must run without an outer autocast context"
            )
        policy_kwargs = {
            key: kwargs[key]
            for key in ("noise", "time")
            if key in kwargs and kwargs[key] is not None
        }
        losses = self.policy_model(
            observation,
            actions.to(torch.float32),
            **policy_kwargs,
        )
        if isinstance(losses, (list, tuple)):
            losses = torch.stack(list(losses))
        return {"action_loss": losses.mean()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: Any) -> dict[str, np.ndarray]:
        chunks: list[np.ndarray] = []
        num_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        device = next(self.policy_model.parameters()).device
        for sample in examples:
            observation, _, transformed = self._prepare_batch([sample], include_actions=False)
            noise = None
            if POLICY_RNG_SEED_KEY in sample:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(sample[POLICY_RNG_SEED_KEY]))
                noise = torch.randn(
                    (1, self.action_horizon, self._model_config.action_dim),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            normalized = self.policy_model.sample_actions(
                device,
                observation,
                noise=noise,
                num_steps=num_steps,
            )
            output = {
                "state": transformed["state"][0].detach().cpu().numpy(),
                "actions": normalized[0].detach().cpu().numpy(),
            }
            output = self._output_transform(output)
            actions = np.asarray(output["actions"], dtype=np.float32)
            if actions.shape != (self.action_horizon, self.environment_action_dim):
                raise ValueError(
                    f"PI0.5 output must be "
                    f"[{self.action_horizon},{self.environment_action_dim}], got {actions.shape}"
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("PI0.5 output contains NaN or Inf")
            chunks.append(actions)
        return {"actions": np.stack(chunks, axis=0)}

    def load_openpi_weights(self, checkpoint: str | os.PathLike[str]) -> None:
        from safetensors.torch import load_model

        path = Path(os.path.expanduser(str(checkpoint)))
        if path.is_dir():
            path = path / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"PI0.5 safetensors checkpoint not found: {path}")
        missing, unexpected = load_model(self.policy_model, str(path), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"PI0.5 checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def load_starvla_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        keys = set(state_dict)
        if keys and all(key.startswith("policy_model.") for key in keys):
            self.load_state_dict(state_dict, strict=True)
            return
        self.policy_model.load_state_dict(state_dict, strict=True)

    def save_preprocessing_assets(self, output_dir: str | os.PathLike[str]) -> None:
        destination = Path(output_dir) / "openpi_assets"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.assets_dir / "norm_stats.json", destination / "norm_stats.json")
