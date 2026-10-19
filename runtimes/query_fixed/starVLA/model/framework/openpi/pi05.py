"""Native StarVLA wrapper around the official OpenPI PyTorch PI0.5 model.

StarVLA owns dataset IO, batching, training, checkpoints, deployment and the
benchmark harness. OpenPI remains the source of truth for the PI0.5 network,
tokenizer, image preprocessing, normalization, loss and action sampling.
"""

from __future__ import annotations

import copy
import dataclasses
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


def _verify_teacher_identity(identity_file: Path, identity: dict, *, is_writer: bool) -> None:
    """Validate immutable resume metadata; publish new metadata atomically."""
    import json
    import tempfile

    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("Fixed initial teacher changed across resume")
        # Every rank can read this file during startup. Never truncate a valid
        # resume identity just to write the same contents again.
        return
    if not is_writer:
        return
    identity_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=identity_file.parent,
                                         prefix=".teacher_identity.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(identity, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, identity_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        foresight_cfg = _cfg_get(pi_cfg, "foresight")
        if foresight_cfg is not None:
            from omegaconf import OmegaConf
            if callable(getattr(foresight_cfg, "unwrap", None)):
                foresight_cfg = foresight_cfg.unwrap()
            if OmegaConf.is_config(foresight_cfg):
                foresight_cfg = OmegaConf.to_container(foresight_cfg, resolve=True)
            elif not isinstance(foresight_cfg, dict):
                foresight_cfg = vars(foresight_cfg)
        self._model_config = dataclasses.replace(self._model_config,
            foresight=foresight_cfg,
            deterministic_images=bool(_cfg_get(pi_cfg, "deterministic_images", False)),
            compile_sample_actions=bool(_cfg_get(pi_cfg, "compile_sample_actions", True)))
        self.initial_checkpoint = str(_cfg_get(pi_cfg, "init_checkpoint"))
        self.run_output_dir = _cfg_get(config, "output_dir", None)
        self._teacher = None
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
        fs = self.policy_model.foresight
        if fs.enabled and fs.augmentation == "official_pytorch_temporal_sync_v1":
            if self._model_config.deterministic_images or not fs.auxiliary_enabled:
                raise ValueError("Synchronized foresight requires train augmentation and an active future branch")
        data_cfg = _cfg_get(_cfg_get(config, "datasets"), "vla_data")
        if fs.enabled and data_cfg is not None:
            if tuple(_cfg_get(data_cfg, "foresight_offsets", fs.future_offsets)) != fs.future_offsets:
                raise ValueError("Model/dataset foresight offsets disagree")
            if tuple(_cfg_get(data_cfg, "foresight_view_keys", fs.future_view_keys)) != fs.future_view_keys:
                raise ValueError("Model/dataset foresight camera order disagrees")
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
            "foresight": {
                "enabled": self.policy_model.foresight.enabled,
                "num_queries": self.policy_model.foresight.num_queries if self.policy_model.foresight.enabled else 0,
                "attention_mode": self.policy_model.foresight.attention_mode,
                "future_offsets": list(self.policy_model.foresight.future_offsets),
                "future_view_keys": list(self.policy_model.foresight.future_view_keys),
                "teacher_at_inference": False,
                "decoder_at_inference": False,
            },
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
            self._input_transform(self._sample_to_openpi(sample, include_actions=include_actions))
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
        if any("current_valid" in sample for sample in examples):
            valid = np.stack([sample.get("current_valid", [True] * len(self.image_observation_keys)) for sample in examples])
            for v, key in enumerate(observation.image_masks):
                observation.image_masks[key] &= torch.as_tensor(valid[:, v], device=device)
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
        auxiliary = self.policy_model.foresight.enabled and self.policy_model.foresight.auxiliary_enabled
        paired_images = None
        if auxiliary:
            fs = self.policy_model.foresight
            for sample in examples:
                meta = sample.get("sample_meta", {})
                if tuple(meta.get("offsets", ())) != fs.future_offsets or tuple(meta.get("camera_order", ())) != fs.future_view_keys:
                    raise ValueError("Future target metadata does not match model time/camera contract")
            if fs.augmentation == "official_pytorch_temporal_sync_v1":
                if any("future_features" in sample for sample in examples):
                    raise ValueError("Synchronized augmentation requires RGB targets, not cached features")
                from openpi.models_pytorch.foresight_preprocessing import preprocess_pair
                paired_images = torch.as_tensor(np.stack([s["future_images"] for s in examples]), device=actions.device)
                observation, paired_images, _ = preprocess_pair(
                    observation, paired_images, camera_order=fs.future_view_keys, train=self.training)
                policy_kwargs["images_preprocessed"] = True
        losses = self.policy_model(
            observation,
            actions.to(torch.float32),
            return_components=auxiliary,
            **policy_kwargs,
        )
        if auxiliary:
            from openpi.models_pytorch.foresight import future_loss_terms
            device = actions.device
            valid = torch.as_tensor(np.stack([s["future_valid"] for s in examples]), device=device, dtype=torch.bool)
            if all("future_features" in s for s in examples):
                teacher = self.get_future_teacher(device)
                for s in examples:
                    if s.get("future_teacher_identity") != teacher.identity:
                        raise ValueError("Precomputed target identity mismatch")
                targets = torch.as_tensor(np.stack([s["future_features"] for s in examples]), device=device)
            else:
                images = paired_images if paired_images is not None else torch.as_tensor(
                    np.stack([s["future_images"] for s in examples]), device=device)
                targets = self.get_future_teacher(device).encode(images, valid)
            total, count, per_head = future_loss_terms(losses["future_prediction"], targets, valid)
            return {"action_loss": losses["action_loss"].mean(), "future_sum": total,
                    "future_count": count, "future_head_sum": (per_head * valid).sum(0),
                    "future_head_count": valid.sum(0)}
        if isinstance(losses, (list, tuple)):
            losses = torch.stack(list(losses))
        return {"action_loss": losses.mean()}

    def get_future_teacher(self, device):
        if self._teacher is None:
            from openpi.models_pytorch.foresight_targets import FrozenInitialTeacher, EMATeacher
            decay = self.policy_model.foresight.teacher_ema_decay
            teacher_class = FrozenInitialTeacher if decay is None else EMATeacher
            options = {} if decay is None else {"decay": decay}
            self._teacher = teacher_class(
                self.policy_model.paligemma_with_expert.paligemma.config.vision_config,
                self.initial_checkpoint, dtype=self._model_config.dtype,
                microbatch=self.policy_model.foresight.teacher_microbatch, **options)
            if self.policy_model.foresight.augmentation != "deterministic":
                self._teacher.identity["preprocessing"] = self.policy_model.foresight.augmentation
            if self.run_output_dir:
                identity_file = Path(self.run_output_dir) / "teacher_identity.json"
                _verify_teacher_identity(identity_file, self._teacher.identity,
                                         is_writer=int(os.environ.get("RANK", "0")) == 0)
        self._teacher.ensure_device(device)
        return self._teacher

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
            # Legacy DeepSpeed/server BF16 casts include the action/time heads.
            # Their inputs remain FP32; use the same autocast contract as train.
            # Native selective-precision policies retain their original path.
            with torch.autocast(device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda" and self.policy_model.action_in_proj.weight.dtype == torch.bfloat16):
                normalized = self.policy_model.sample_actions(
                    device, observation, noise=noise, num_steps=num_steps)
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
        enabled = self.policy_model.foresight.enabled
        missing, unexpected = load_model(self.policy_model, str(path), strict=not enabled)
        allowed = {name for name in self.policy_model.state_dict()
                   if name == "future_queries" or name == "foresight_signature" or name.startswith("future_decoder.")}
        if unexpected or (set(missing) != allowed if enabled else bool(missing)):
            raise RuntimeError(f"PI0.5 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if enabled:
            print("FORESIGHT_INITIAL_MISSING_KEYS", sorted(missing))

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
        if self.policy_model.foresight.enabled:
            from omegaconf import OmegaConf
            p = self.policy_model
            OmegaConf.save(OmegaConf.create({
                "foresight": dataclasses.asdict(p.foresight),
                "resolved_shapes": {"D": p.future_queries.shape[-1], "Dv": p.foresight_dv,
                    "grid": list(p.foresight_grid), "queries": list(p.future_queries.shape),
                    "target": ["B", 2, len(p.foresight.future_view_keys),
                               p.foresight_grid[0] * p.foresight_grid[1], p.foresight_dv]},
                "visibility": [[1,int(p.foresight.attention_mode == "bidirectional_xq"),0],[1,1,0],[1,1,1]],
                "initial_checkpoint": self.initial_checkpoint,
                "preprocessing": (p.foresight.augmentation if p.foresight.augmentation != "deterministic"
                                  else "openpi_resize_with_pad_rgb_224_minus1_plus1_v1"),
            }), Path(output_dir) / "resolved_config.yaml")
