"""Fixed initial SigLIP teacher and identity-checked per-frame target cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import SiglipVisionModel

from openpi.models_pytorch.foresight import tensor_state_hash
from openpi.models_pytorch.foresight_ema import MomentumState, ensure_model_device


class FrozenInitialTeacher:
    """Not an nn.Module child: never optimized, autocast/DS-cast, or serialized as student."""
    def __init__(self, vision_config, checkpoint, *, dtype="bfloat16", microbatch=4):
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        with torch.random.fork_rng(devices=[]):
            self.model = SiglipVisionModel(vision_config)
        if dtype == "bfloat16":
            self.model.bfloat16()
            for name, p in self.model.named_parameters():
                if ("embeddings.patch_embedding" in name or "embeddings.position_embedding" in name
                        or "input_layernorm" in name or "post_attention_layernorm" in name or "model.norm" in name):
                    p.data = p.data.float()
        prefix = "paligemma_with_expert.paligemma.model.vision_tower."
        with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
            weights = {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
            if not weights:
                prefix = "paligemma_with_expert.paligemma.vision_tower."
                weights = {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
        self.model.load_state_dict(weights, strict=True)
        self.model.requires_grad_(False).eval()
        self.model.gradient_checkpointing_disable()
        self.microbatch = int(microbatch)
        self.dv = vision_config.hidden_size
        self.grid = (vision_config.image_size // vision_config.patch_size,) * 2
        self.identity = {
            "teacher_checkpoint": str(checkpoint.resolve()),
            "teacher_sha256": tensor_state_hash(self.model),
            "preprocessing": "openpi_resize_with_pad_rgb_224_minus1_plus1_v1",
            "target_space": "pre_projector_last_hidden_state",
            "grid": list(self.grid), "dv": self.dv, "precision": dtype,
        }

    def ensure_device(self, device):
        return ensure_model_device(self.model, device)

    @torch.no_grad()
    def encode(self, images, valid):
        self.ensure_device(images.device)
        if images.shape[:3] != valid.shape or images.shape[-3:] != (3, 224, 224):
            raise ValueError("Invalid future image contract")
        if not torch.isfinite(images).all():
            raise FloatingPointError("Nonfinite teacher images")
        flat = images.reshape(-1, 3, 224, 224)
        ids = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        target = torch.zeros(flat.shape[0], self.grid[0] * self.grid[1], self.dv,
                             device=images.device, dtype=torch.float32)
        with torch.autocast(images.device.type, enabled=False):
            for chunk in ids.split(self.microbatch):
                if chunk.numel():
                    z = self.model(pixel_values=flat[chunk].float(), return_dict=True).last_hidden_state
                    target.index_copy_(0, chunk, z.float())
        return target.reshape(*valid.shape, self.grid[0] * self.grid[1], self.dv)


class EMATeacher(MomentumState, FrozenInitialTeacher):
    def __init__(self, *args, decay, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_momentum(decay)


class TargetCache:
    """Optional sparse cache; manifest mismatch fails before any feature is read."""
    def __init__(self, directory, expected_manifest, *, create=False):
        self.root = Path(directory)
        required = {"data_version", "split", "camera_order", "teacher_checkpoint",
                    "teacher_checkpoint_sha256", "teacher_sha256", "preprocessing", "grid", "dv"}
        if required - set(expected_manifest):
            raise ValueError(f"Incomplete target cache manifest: {sorted(required - set(expected_manifest))}")
        self.manifest = {**expected_manifest, "storage_dtype": "float16", "cache_format": "per_frame_v1"}
        file = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not file.exists():
                file.write_text(json.dumps(self.manifest, sort_keys=True, indent=2))
        if not file.exists() or json.loads(file.read_text()) != self.manifest:
            raise ValueError("Future target cache manifest mismatch")

    def _path(self, dataset, episode, frame, camera):
        key = json.dumps([dataset, int(episode), int(frame), camera], separators=(",", ":"))
        return self.root / (hashlib.sha256(key.encode()).hexdigest() + ".pt")

    def write(self, dataset, episode, frame, camera, feature):
        torch.save({"key": [dataset, int(episode), int(frame), camera],
                    "feature": feature.detach().cpu().half()}, self._path(dataset, episode, frame, camera))

    def read(self, dataset, episode, frame, camera):
        item = torch.load(self._path(dataset, episode, frame, camera), weights_only=True, map_location="cpu")
        if item["key"] != [dataset, int(episode), int(frame), camera]:
            raise ValueError("Target cache episode/frame/camera key mismatch")
        value = item["feature"]
        if list(value.shape) != [self.manifest["grid"][0] * self.manifest["grid"][1], self.manifest["dv"]]:
            raise ValueError("Future target cache shape mismatch")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite cached target")
        return value.float()
