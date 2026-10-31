"""Small state helpers used by the explicit training loop."""

from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn


class TrainingProgress:
    """Optimization and dataloader progress persisted by Accelerate."""

    def __init__(self, completed_steps: int = 0, consumed_batches: int = 0):
        self.completed_steps = int(completed_steps)
        self.consumed_batches = int(consumed_batches)

    def state_dict(self) -> dict[str, int]:
        return {
            "completed_steps": self.completed_steps,
            "consumed_batches": self.consumed_batches,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.completed_steps = int(state_dict["completed_steps"])
        self.consumed_batches = int(state_dict["consumed_batches"])

    def dataloader_position(self, batches_per_epoch: int) -> tuple[int, int]:
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        return divmod(self.consumed_batches, batches_per_epoch)


class ExponentialMovingAverage:
    """FP32 parameter EMA with tied-weight aware serialization.

    OpenPI's JAX trainer keeps trainable parameters and EMA parameters in full
    precision while using bfloat16 for the expensive model compute.  PI0's
    PyTorch module stores most model parameters as bfloat16, so cloning the
    model dtype would accumulate the EMA in bfloat16.  Shadows here are always
    float32 and remain float32 in the EMA checkpoint.

    ZeRO-2 keeps model parameters replicated, allowing each rank to maintain
    an identical device-local shadow while only rank zero serializes it.
    """

    def __init__(self, decay: float = 0.99):
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.alias_to_primary: dict[str, str] = {}
        self.shadow_parameters: dict[str, torch.Tensor] = {}

    @property
    def is_initialized(self) -> bool:
        return bool(self.shadow_parameters)

    @staticmethod
    def _named_parameters_with_duplicates(model: nn.Module):
        try:
            return model.named_parameters(remove_duplicate=False)
        except TypeError:  # pragma: no cover - compatibility with old torch.
            return model.named_parameters()

    def initialize(self, model: nn.Module) -> None:
        """Initialize FP32 EMA from the model's current weights."""
        self.alias_to_primary = {}
        self.shadow_parameters = {}
        object_id_to_primary: dict[int, str] = {}

        with torch.no_grad():
            for name, parameter in self._named_parameters_with_duplicates(model):
                object_id = id(parameter)
                primary = object_id_to_primary.get(object_id)
                if primary is None:
                    primary = name
                    object_id_to_primary[object_id] = primary
                    self.shadow_parameters[primary] = (
                        parameter.detach().to(dtype=torch.float32).clone()
                    )
                self.alias_to_primary[name] = primary

        if not self.shadow_parameters:
            raise ValueError("Cannot initialize EMA from a model without parameters")
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one FP32 ``ema = decay * ema + (1-decay) * parameter`` update."""
        if not self.is_initialized:
            raise RuntimeError("EMA must be initialized before update()")

        current_parameters: dict[str, torch.Tensor] = {}
        seen_aliases: set[str] = set()
        for name, parameter in self._named_parameters_with_duplicates(model):
            seen_aliases.add(name)
            primary = self.alias_to_primary.get(name)
            if primary is None:
                raise RuntimeError(f"Model parameter topology changed after EMA initialization: {name}")
            current_parameters.setdefault(primary, parameter)

        missing_aliases = set(self.alias_to_primary) - seen_aliases
        if missing_aliases:
            raise RuntimeError(
                "Model parameter topology changed after EMA initialization; "
                f"missing aliases={sorted(missing_aliases)}"
            )

        one_minus_decay = 1.0 - self.decay
        for primary, shadow in self.shadow_parameters.items():
            parameter = current_parameters[primary].detach().to(
                device=shadow.device, dtype=torch.float32
            )
            shadow.mul_(self.decay).add_(parameter, alpha=one_minus_decay)
        self.num_updates += 1

    def export_state_dict(self, model_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return a CPU state dict with FP32 EMA parameters and current buffers.

        Every alias is cloned independently because safetensors rejects shared
        storage. Non-parameter buffers retain their current dtype and value.
        """
        if not self.is_initialized:
            raise RuntimeError("EMA must be initialized before export")

        missing = set(self.alias_to_primary) - set(model_state_dict)
        if missing:
            raise KeyError(f"Model state dict is missing EMA parameter aliases: {sorted(missing)}")

        exported: dict[str, torch.Tensor] = {}
        for name, value in model_state_dict.items():
            primary = self.alias_to_primary.get(name)
            source = self.shadow_parameters[primary] if primary is not None else value.detach()
            exported[name] = source.to(device="cpu").contiguous().clone()
        return exported

    def save_safetensors(self, path: str | Path, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Save a strict-loadable full state with FP32 EMA parameters."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            self.export_state_dict(model_state_dict),
            str(path),
            metadata={
                "ema_decay": repr(self.decay),
                "ema_num_updates": str(self.num_updates),
                "ema_dtype": "float32",
                "weight_role": "ema_inference",
                "ema_aliases": json.dumps(self.alias_to_primary, sort_keys=True),
            },
        )

    @torch.no_grad()
    def load_safetensors(self, path: str | Path) -> None:
        """Restore FP32 EMA shadows from a full model safetensors checkpoint."""
        from safetensors import safe_open
        from safetensors.torch import load_file

        if not self.is_initialized:
            raise RuntimeError("EMA must be initialized from the model before loading")

        path = Path(path)
        state_dict = load_file(str(path), device="cpu")
        primary_to_aliases: dict[str, list[str]] = {}
        for alias, primary in self.alias_to_primary.items():
            primary_to_aliases.setdefault(primary, []).append(alias)

        missing: list[str] = []
        for primary, shadow in self.shadow_parameters.items():
            aliases = primary_to_aliases[primary]
            source_name = primary if primary in state_dict else next(
                (alias for alias in aliases if alias in state_dict), None
            )
            if source_name is None:
                missing.append(primary)
                continue
            source = state_dict[source_name]
            if source.dtype != torch.float32 or source.shape != shadow.shape or not torch.isfinite(source).all():
                raise ValueError(f'Invalid FP32 EMA parameter: {source_name}')
            for alias in aliases:
                if alias not in state_dict or not torch.equal(source, state_dict[alias]):
                    raise ValueError(f'EMA tied alias missing/different: {alias}')
            shadow.copy_(state_dict[source_name].to(device=shadow.device, dtype=torch.float32))
            if not torch.equal(shadow, source.to(device=shadow.device)):
                raise ValueError(f'EMA restore differs: {source_name}')
        if missing:
            raise KeyError(f"EMA checkpoint is missing parameters: {sorted(missing)}")

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if json.loads(metadata.get('ema_aliases', '{}')) != self.alias_to_primary:
            raise ValueError('EMA saved parameter aliases differ from current model')
        stored_decay = float(metadata.get("ema_decay", self.decay))
        if stored_decay != self.decay:
            raise ValueError(
                f"EMA decay mismatch: checkpoint={stored_decay}, configured={self.decay}"
            )
        stored_dtype = metadata.get("ema_dtype", "float32")
        if stored_dtype != "float32":
            raise ValueError(f"EMA checkpoint must contain FP32 shadows, got {stored_dtype!r}")
        self.num_updates = int(metadata.get("ema_num_updates", 0))

def clone_shared_tensors_for_safetensors(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone only secondary storage aliases so save_file accepts the state.

    PaliGemma ties its token embedding and LM-head weights. Accelerate's model
    state preserves that alias, while safetensors save_file intentionally
    rejects shared storage. Keeping the first tensor and cloning later aliases
    retains every strict-load key without cloning the complete multi-GB state.
    """
    safe_state: dict[str, torch.Tensor] = {}
    seen_storages: set[tuple[str, int]] = set()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor state for {name!r}, got {type(tensor)!r}")
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr())
        if storage.nbytes() and storage_key in seen_storages:
            safe_state[name] = tensor.detach().clone()
        else:
            safe_state[name] = tensor
            if storage.nbytes():
                seen_storages.add(storage_key)
    return safe_state
