# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
from contextlib import nullcontext
import json
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
from transformers.trainer_utils import get_last_checkpoint

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.checkpoint_state import (
    ExponentialMovingAverage,
    TrainingProgress,
    clone_shared_tensors_for_safetensors,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def pi05_official_precision_enabled(cfg) -> bool:
    """Return whether PI0.5 should retain OpenPI's native mixed parameter dtypes."""
    framework_cfg = getattr(cfg, "framework", None)
    pi05_cfg = getattr(framework_cfg, "pi05", None)
    return bool(getattr(pi05_cfg, "official_precision_mode", False))


def build_accelerator(cfg) -> Accelerator:
    """Construct Accelerate after YAML parsing so accumulation is effective."""
    gradient_accumulation_steps = int(
        getattr(cfg.trainer, "gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("trainer.gradient_accumulation_steps must be positive")

    official_precision = pi05_official_precision_enabled(cfg)
    if official_precision:
        # DeepSpeed calls module.bfloat16(), destroying OpenPI's deliberate
        # FP32 AdaRMS/LayerNorm, vision embedding, and action/time-head islands.
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
            raise RuntimeError(
                "framework.pi05.official_precision_mode=true requires native DDP. "
                "Use run_pi05_robocasa365.sh (it selects the DDP config automatically) "
                "or pass accelerate_official_precision_ddp.yaml explicitly."
            )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size >= 8:
            # Match the pinned upstream train_pytorch.py resource branch.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )
        from pri4r_native_state import accumulation_plugin
        accelerator = Accelerator(
            gradient_accumulation_plugin=accumulation_plugin(gradient_accumulation_steps),
            mixed_precision="no",
            kwargs_handlers=[ddp_kwargs],
        )
    else:
        # Preserve the historical StarVLA path when the opt-in mode is disabled.
        accelerator = Accelerator(
            deepspeed_plugin=DeepSpeedPlugin(),
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=gradient_accumulation_steps, sync_each_batch=True),
        )
    accelerator.print(accelerator.state)
    if official_precision:
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise RuntimeError("Official PI0.5 precision mode cannot run through DeepSpeed")
        if accelerator.mixed_precision != "no":
            raise RuntimeError(
                "Official PI0.5 precision mode requires Accelerator mixed_precision='no', "
                f"got {accelerator.mixed_precision!r}"
            )
    if accelerator.gradient_accumulation_steps != gradient_accumulation_steps:
        raise RuntimeError(
            "Accelerate ignored trainer.gradient_accumulation_steps: "
            f"configured={gradient_accumulation_steps}, "
            f"effective={accelerator.gradient_accumulation_steps}"
        )
    return accelerator


def _link_or_copy(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Expose one checkpoint at a second path without duplicating it when possible."""
    source = os.fspath(source)
    destination = os.fspath(destination)
    try:
        os.link(source, destination)
    except FileExistsError:
        if not os.path.samefile(source, destination):
            shutil.copy2(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    official = pi05_official_precision_enabled(cfg)
    if official:
        # Official train_pytorch.py uses one AdamW group over model.parameters().
        param_groups = [{"params": list(model.parameters()), "name": "openpi_all",
                         "lr": cfg.trainer.learning_rate.base}]
    foresight = getattr(getattr(model, "policy_model", None), "foresight", None)
    if not official and foresight is not None and (foresight.enabled or getattr(getattr(model.policy_model, "point_track", None), "enabled", False) or getattr(model.policy_model.config, "deterministic_images", False)):
        names = {id(p): name for name, p in model.named_parameters()}
        split_groups = []
        for group in param_groups:
            for no_decay in (False, True):
                params = [p for p in group["params"] if p.requires_grad and
                          (p.ndim < 2 or names[id(p)].endswith(("future_queries", "embedding.queries"))) == no_decay]
                if params:
                    split_groups.append({**group, "params": params,
                        "name": group["name"] + ("_no_decay" if no_decay else "_decay"),
                        "weight_decay": 0. if no_decay else cfg.trainer.optimizer.weight_decay})
        param_groups = split_groups
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=bool(cfg.trainer.optimizer.get("fused", True)),
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    if cfg.trainer.lr_scheduler_type == "openpi_cosine":
        # Match scripts/train_pytorch.py in the pinned OpenPI checkout exactly:
        # warm up from peak_lr / (warmup_steps + 1), then cosine decay to
        # min_lr over the full configured training-step interval.
        peak_lr = float(cfg.trainer.learning_rate.base)
        warmup_steps = int(cfg.trainer.num_warmup_steps)
        decay_steps = int(cfg.trainer.max_train_steps)
        min_lr = float(cfg.trainer.scheduler_specific_kwargs.min_lr)

        def openpi_lr_lambda(step: int) -> float:
            if step < warmup_steps:
                initial_lr = peak_lr / (warmup_steps + 1)
                value = initial_lr + (peak_lr - initial_lr) * step / max(1, warmup_steps)
            else:
                progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                value = min_lr + (peak_lr - min_lr) * cosine
            return value / peak_lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, openpi_lr_lambda)
    else:
        # Strip keys unknown to transformers' get_scheduler before passing kwargs.
        sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
        lr_scheduler = get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=sched_kwargs,
        )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.progress = TrainingProgress()
        self.ema: ExponentialMovingAverage | None = None
        self.vla_epoch_count = 0
        self.total_batch_size = self._calculate_total_batch_size()

    @property
    def completed_steps(self):
        return self.progress.completed_steps

    @completed_steps.setter
    def completed_steps(self, value):
        self.progress.completed_steps = int(value)

    @property
    def consumed_batches(self):
        return self.progress.consumed_batches

    @consumed_batches.setter
    def consumed_batches(self, value):
        self.progress.consumed_batches = int(value)

    def prepare_training(self):
        expected_global_batch_size = getattr(
            self.config.trainer, "expected_global_batch_size", None
        )
        if (
            expected_global_batch_size is not None
            and self.total_batch_size != int(expected_global_batch_size)
        ):
            raise ValueError(
                "Global batch size does not match the JAX parity contract: "
                f"effective={self.total_batch_size}, "
                f"expected={int(expected_global_batch_size)}. "
                "Check per_device_batch_size, world size, and gradient accumulation."
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self.accelerator.register_for_checkpointing(self.lr_scheduler, self.progress)
        self._register_auxiliary_checkpoints()

        if self.resume_from_checkpoint:
            self._load_checkpoint(self.resume_from_checkpoint)
        self._validate_framework_precision()
        self._initialize_ema()

    def _register_auxiliary_checkpoints(self):
        """Register framework state after scheduler/progress, before full-state restore."""

    def _validate_framework_precision(self):
        """Run a framework-owned post-prepare dtype assertion when requested."""
        if not pi05_official_precision_enabled(self.config):
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        validate = getattr(unwrapped_model, "validate_training_precision", None)
        if not callable(validate):
            raise AttributeError(
                "Official PI0.5 precision mode requires validate_training_precision() "
                "on the selected framework"
            )
        report = validate()
        self.accelerator.print(f"PI0.5 official precision validated: {report}")

    def _initialize_ema(self):
        """Initialize or restore the JAX-style FP32 inference EMA."""
        configured_decay = getattr(self.config.trainer, "ema_decay", None)
        if configured_decay is None:
            return
        decay = float(configured_decay)
        if decay <= 0.0:
            return
        if getattr(self.config.trainer, "save_format", "pt") != "safetensors":
            raise ValueError("trainer.ema_decay requires trainer.save_format=safetensors")

        self.ema = ExponentialMovingAverage(decay=decay, device=self.config.trainer.get('ema_device',None))
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema.initialize(unwrapped_model)

        if self.resume_from_checkpoint:
            ema_checkpoint = Path(self.resume_from_checkpoint) / "ema_model.safetensors"
            if ema_checkpoint.is_file():
                self.ema.load_safetensors(ema_checkpoint)
                logger.info(
                    "Restored FP32 EMA from %s (decay=%s, updates=%s)",
                    ema_checkpoint,
                    self.ema.decay,
                    self.ema.num_updates,
                )
            else:
                missing_policy = str(
                    getattr(self.config.trainer, "ema_missing_resume_policy", "error")
                )
                if missing_policy == "initialize":
                    logger.warning(
                        "Resume checkpoint %s predates EMA; initialized FP32 EMA from "
                        "its raw model weights.",
                        self.resume_from_checkpoint,
                    )
                elif missing_policy == "error":
                    raise FileNotFoundError(
                        f"EMA is enabled but resume checkpoint lacks {ema_checkpoint}. "
                        "Set trainer.ema_missing_resume_policy=initialize only for a "
                        "deliberate transition from a legacy raw-only checkpoint."
                    )
                else:
                    raise ValueError(
                        "trainer.ema_missing_resume_policy must be 'error' or 'initialize', "
                        f"got {missing_policy!r}"
                    )
        logger.info(
            "Enabled PI0.5 FP32 inference EMA (decay=%s, save_raw=%s)",
            self.ema.decay,
            bool(getattr(self.config.trainer, "ema_save_raw_model", False)),
        )

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        asset_hook = getattr(self.model, "save_preprocessing_assets", None)
        if callable(asset_hook):
            asset_hook(output_dir)
            logger.info("Saved framework preprocessing assets at %s", output_dir)

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Resolve a full-state resume checkpoint or load model-only warm-start weights."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = None

        if is_resume:
            resume_from_checkpoint = getattr(self.config.trainer, "resume_from_checkpoint", None)
            if resume_from_checkpoint:
                resume_from_checkpoint = os.path.expanduser(str(resume_from_checkpoint))
            else:
                resume_from_checkpoint = get_last_checkpoint(self.checkpoint_dir)

            if resume_from_checkpoint:
                if not os.path.isdir(resume_from_checkpoint):
                    raise FileNotFoundError(f"Resume checkpoint is not a directory: {resume_from_checkpoint}")
                self.resume_from_checkpoint = resume_from_checkpoint
                logger.info(f"Full-state checkpoint selected for resume: {self.resume_from_checkpoint}")
                return

            raise FileNotFoundError(
                f"trainer.is_resume=true, but no checkpoint-* directory was found in {self.checkpoint_dir}"
            )

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")

    @staticmethod
    def _trusted_checkpoint_safe_globals():
        """Allow NumPy scalar metadata stored by our own optimizer checkpoints."""
        safe_globals = [np.core.multiarray.scalar, np.dtype]
        numpy_dtypes = getattr(np, "dtypes", None)
        if numpy_dtypes is not None:
            float64_dtype = getattr(numpy_dtypes, "Float64DType", None)
            if float64_dtype is not None:
                safe_globals.append(float64_dtype)
        return torch.serialization.safe_globals(safe_globals)

    def _load_checkpoint(self, checkpoint_path):
        """Restore model, optimizer, scheduler, RNG, and registered progress state."""
        self.accelerator.wait_for_everyone()
        # PyTorch 2.6+ defaults torch.load to weights_only=True. Our trusted
        # optimizer checkpoints contain NumPy scalar LR metadata.
        with self._trusted_checkpoint_safe_globals():
            self.accelerator.load_state(checkpoint_path)
        if pi05_official_precision_enabled(self.config):
            from pri4r_native_state import restore_rng_strict
            restore_rng_strict(checkpoint_path,self.accelerator.process_index)
        self.accelerator.wait_for_everyone()
        self.accelerator.print(
            f"Resumed from checkpoint: {checkpoint_path} "
            f"(completed_steps={self.completed_steps}, consumed_batches={self.consumed_batches})"
        )

    def _save_checkpoint(self):
        """Save a resumable state directory and a model-only file for evaluation."""
        state_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{self.completed_steps}")
        self.accelerator.wait_for_everyone()
        # DeepSpeed checkpointing is collective: every rank must enter save_state.
        self.accelerator.save_state(state_checkpoint_path)
        self.accelerator.wait_for_everyone()

        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            model_checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            evaluation_weights_path = None
            raw_weights_path = None
            ema_state_path = None

            if save_format == "safetensors":
                from safetensors.torch import save_file

                if self.ema is None:
                    evaluation_weights_path = model_checkpoint_path + "_model.safetensors"
                    save_file(
                        clone_shared_tensors_for_safetensors(state_dict),
                        evaluation_weights_path,
                    )
                else:
                    if bool(getattr(self.config.trainer, "ema_save_raw_model", False)):
                        raw_weights_path = model_checkpoint_path + "_raw_model.safetensors"
                        save_file(
                            clone_shared_tensors_for_safetensors(state_dict),
                            raw_weights_path,
                        )
                    ema_state_path = os.path.join(state_checkpoint_path, "ema_model.safetensors")
                    self.ema.save_safetensors(ema_state_path, state_dict)
                    evaluation_weights_path = model_checkpoint_path + "_model.safetensors"
                    _link_or_copy(ema_state_path, evaluation_weights_path)
            elif save_format == "pt":
                if self.ema is not None:
                    raise ValueError("EMA model-only checkpoints require safetensors")
                evaluation_weights_path = model_checkpoint_path + "_pytorch_model.pt"
                torch.save(state_dict, evaluation_weights_path)
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {
                "steps": self.completed_steps,
                "consumed_batches": self.consumed_batches,
                "state_checkpoint": state_checkpoint_path,
                "evaluation_weights": evaluation_weights_path,
                "raw_weights": raw_weights_path,
                "ema_state": ema_state_path,
                "ema_decay": self.ema.decay if self.ema is not None else None,
                "ema_updates": self.ema.num_updates if self.ema is not None else None,
                "ema_dtype": "float32" if self.ema is not None else None,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Resumable checkpoint saved at {state_checkpoint_path}")
            self.accelerator.print(f"✅ Evaluation weights saved at {evaluation_weights_path}")

            if raw_weights_path is not None:
                self.accelerator.print(f"Raw comparison weights saved at {raw_weights_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.consumed_batches / len(self.vla_train_dataloader), 2)
            metrics["consumed_batches"] = self.consumed_batches
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _set_dataloader_epoch(self, epoch):
        """Keep sampler and deterministic mixture-dataset epoch in sync."""
        seen = set()
        sampler_candidates = [
            getattr(self.vla_train_dataloader, "sampler", None),
            getattr(getattr(self.vla_train_dataloader, "batch_sampler", None), "sampler", None),
        ]
        for sampler in sampler_candidates:
            if sampler is not None and id(sampler) not in seen:
                seen.add(id(sampler))
                set_epoch = getattr(sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)

        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _create_data_iterators(self):
        """Create the iterator at the exact consumed-batch position."""
        batches_per_epoch = len(self.vla_train_dataloader)
        self.vla_epoch_count, batches_to_skip = self.progress.dataloader_position(batches_per_epoch)
        self._set_dataloader_epoch(self.vla_epoch_count)

        if batches_to_skip:
            resume_dataloader = self.accelerator.skip_first_batches(
                self.vla_train_dataloader, num_batches=batches_to_skip
            )
            self.vla_iter = iter(resume_dataloader)
        else:
            self.vla_iter = iter(self.vla_train_dataloader)

        if self.consumed_batches:
            self.accelerator.print(
                f"Dataloader resumed after {self.consumed_batches} consumed batches "
                f"(epoch={self.vla_epoch_count}, skipped_in_epoch={batches_to_skip})"
            )

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            self._set_dataloader_epoch(self.vla_epoch_count)
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)

        self.consumed_batches += 1
        return batch_vla

    def train(self):
        """Execute training loop."""
        # Start fork-based DataLoader workers before wandb.init(). W&B installs
        # post-import telemetry hooks backed by an asyncio thread; inheriting
        # those hooks in a forked worker makes the worker fail as soon as it
        # lazily imports a monitored package (for example Hugging Face
        # datasets). Creating the iterator first preserves fork's shared-memory
        # behavior while keeping W&B state entirely in the training process.
        self._create_data_iterators()
        self._init_wandb()
        self._log_training_config()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        stop_after = min(self.config.trainer.max_train_steps,
                         int(self.config.trainer.get("stop_after_steps", self.config.trainer.max_train_steps)))
        while self.completed_steps < stop_after:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if not self.accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics.setdefault("timing/data", t_end_data - t_start_data)
            step_metrics.setdefault("timing/model", t_end_model - t_start_model)
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= stop_after:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            predicted_actions = output_dict.get("actions", output_dict.get("normalized_actions"))
            if predicted_actions is None:
                raise KeyError("Framework evaluation response has neither 'actions' nor 'normalized_actions'")
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(predicted_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(
                "  EMA decay/dtype = %s/%s",
                self.ema.decay if self.ema is not None else None,
                "float32" if self.ema is not None else None,
            )

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            precision_context = (
                nullcontext()
                if pi05_official_precision_enabled(self.config)
                else torch.autocast("cuda", dtype=torch.bfloat16)
            )
            with precision_context:
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            if bool(getattr(self.config.trainer, "fail_on_nonfinite_loss", True)):
                finite_flag = torch.isfinite(total_loss.detach()).all().to(dtype=torch.int32)
                if dist.is_initialized():
                    # Make every rank take the same failure path. Otherwise one
                    # rank may raise while its peers enter backward and hang.
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                if not bool(finite_flag.item()):
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    raise FloatingPointError(
                        f"Non-finite training loss before backward at "
                        f"step={self.completed_steps}, rank={rank}, "
                        f"action_loss={action_loss.detach().float().item()}"
                    )

            self.accelerator.backward(total_loss)

            if (
                self.accelerator.sync_gradients
                and self.config.trainer.gradient_clipping is not None
            ):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                if self.ema is not None:
                    self.ema.update(self.accelerator.unwrap_model(self.model))
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                if self.ema is None:
                    save_file(
                        clone_shared_tensors_for_safetensors(state_dict),
                        os.path.join(final_checkpoint, "model.safetensors"),
                    )
                else:
                    if bool(getattr(self.config.trainer, "ema_save_raw_model", False)):
                        save_file(
                            clone_shared_tensors_for_safetensors(state_dict),
                            os.path.join(final_checkpoint, "raw_model.safetensors"),
                        )
                    self.ema.save_safetensors(
                        os.path.join(final_checkpoint, "model.safetensors"), state_dict
                    )
            elif save_format == "pt":
                if self.ema is not None:
                    raise ValueError("EMA final checkpoints require safetensors")
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    if bool(cfg.trainer.get("deterministic_algorithms", False)):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    accelerator = build_accelerator(cfg)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    existing = Path(cfg.run_root_dir) / cfg.run_id / "config.full.yaml"
    if (cfg.framework.name == "OpenPIPi05" and cfg.framework.pi05.get("deterministic_images", False)
            and existing.exists() and not cfg.trainer.is_resume):
        raise FileExistsError(f"Refusing to overwrite an existing controlled run: {existing}. Choose a new run_id or explicitly resume a complete checkpoint.")
    output_dir = setup_directories(cfg=cfg)
    set_seed(int(cfg.seed))  # New Q/decoder parameters must be reproducible before DDP setup.
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer_class = VLATrainer
    foresight = getattr(getattr(vla, "policy_model", None), "foresight", None)
    if foresight is not None and foresight.enabled and foresight.auxiliary_enabled:
        from starVLA.training.foresight_trainer import ForesightTrainer
        trainer_class = ForesightTrainer
    elif getattr(getattr(vla, "policy_model", None), "point_track", None) is not None and vla.policy_model.point_track.enabled:
        from starVLA.training.query_point_trainer import QueryPointTrainer
        trainer_class = QueryPointTrainer
    trainer = trainer_class(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
