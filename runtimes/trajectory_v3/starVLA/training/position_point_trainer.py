"""V3 Embedding A/B trainer; native full-state saving and update clocks."""
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.distributed as dist

from starVLA.training.query_point_trainer import QueryPointTrainer
from starVLA.training.train_starvla import VLATrainer, pi05_official_precision_enabled


class PositionPointTrainer(QueryPointTrainer):
    def prepare_training(self):
        p = self.model.policy_model
        cfg = p.point_track
        if (not cfg.enabled or cfg.version != "pri4r_embedding_v1"
                or cfg.objective_version != "relative_position_and_difference_v3"):
            raise ValueError("PositionPointTrainer requires the Embedding V3 objective")
        if (p.foresight.enabled or p.foresight.auxiliary_enabled or hasattr(p, "future_queries")
                or getattr(self.model, "_teacher", None) is not None):
            raise ValueError("V3 preserves the original action prefix and has no teacher")
        if not (cfg.camera_role_augmentation and cfg.prefix_cache_alignment):
            raise ValueError("Accepted augmentation/cache alignment must remain enabled")
        ids = {id(x) for group in self.optimizer.param_groups for x in group["params"]}
        required = {n: x for n, x in p.named_parameters() if n.startswith("point_track_head.")}
        if not required or not all(x.requires_grad and id(x) in ids for x in required.values()):
            raise ValueError("Embedding and all point/temporal parameters must be optimized")
        VLATrainer.prepare_training(self)
        if self.resume_from_checkpoint:
            from starVLA.training.foresight_resume_audit import verify_loaded_partition
            verify_loaded_partition(self.model, self.resume_from_checkpoint, self.config.output_dir)
        if self.accelerator.is_main_process:
            runtime = dict(gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           torch=str(torch.__version__), world_size=self.accelerator.num_processes,
                           global_batch=self.total_batch_size, teacher=False, future_queries=False,
                           embedding_query_shape=list(p.point_track_head.embedding.queries.shape),
                           objective=cfg.objective_version, output_kind=cfg.output_kind,
                           temporal_layers=cfg.temporal_layers, temporal_scope="same_point_time",
                           added_rope=False)
            (Path(self.config.output_dir) / "runtime_hardware.json").write_text(json.dumps(runtime, indent=2))
            self.accelerator.print("POSITION_POINT_RUNTIME", runtime)

    def _get_next_batch(self):
        if not hasattr(self, "_pending_batches"):
            self._pending_batches = deque()
        if not self._pending_batches:
            start = time.perf_counter()
            for _ in range(self.accelerator.gradient_accumulation_steps):
                self._pending_batches.append(VLATrainer._get_next_batch(self))
            pos_count, diff_count = 0, 0
            for batch in self._pending_batches:
                for sample in batch:
                    valid = np.asarray(sample["point_valid"])
                    if valid.dtype != np.bool_ or valid.ndim != 2:
                        raise ValueError("Expected bool [N,H] position mask")
                    if not np.array_equal(valid, np.logical_and.accumulate(valid, axis=1)):
                        raise ValueError("Position validity crosses an invalid gap")
                    previous = np.concatenate((np.ones_like(valid[:, :1]), valid[:, :-1]), axis=1)
                    pos_count += 3 * int(valid.sum())
                    diff_count += 3 * int((valid & previous).sum())
            self._global_counts = torch.tensor([pos_count, diff_count], dtype=torch.float32,
                                               device=self.accelerator.device)
            if dist.is_initialized():
                dist.all_reduce(self._global_counts)
            self._step_sums = None
            self._window_data_time = time.perf_counter() - start
            self._window_started = None
        return self._pending_batches.popleft()

    def _train_step(self, batch_vla, batch_vlm=None):
        from openpi.models_pytorch.point_track import distributed_point_track_loss
        from openpi.models_pytorch.position_supervision import diagnostic_names
        from openpi.models_pytorch.trajectory_supervision import auxiliary_ramp
        if self._window_started is None:
            self._window_started = time.perf_counter()
        before, norm = {}, None
        with self.accelerator.accumulate(self.model):
            module = self.accelerator.unwrap_model(self.model)
            cfg = module.policy_model.point_track
            probe = self.completed_steps < 5 or (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0
            if self.accelerator.sync_gradients and probe:
                # Small representatives from every Embedding/temporal block.
                before = {name: p.detach().float().clone() for name, p in module.policy_model.named_parameters()
                          if name.startswith("point_track_head.") and (
                              name.endswith(("queries", "fusion_out.weight", "point_mlp.0.weight",
                                             "self_norm.weight", "cross_norm.weight",
                                             "attention_norm.weight", "ff_norm.weight")))}
            context = nullcontext() if pi05_official_precision_enabled(self.config) else torch.autocast("cuda", dtype=torch.bfloat16)
            with context:
                output = self.model(batch_vla, point_diagnostics=probe)
                component_sums = torch.stack([output["position_sum"], output["difference_sum"]])
                components = distributed_point_track_loss(
                    component_sums, self._global_counts, world_size=self.accelerator.num_processes,
                    accumulation_steps=self.accelerator.gradient_accumulation_steps)
                ramp = auxiliary_ramp(self.completed_steps, cfg.auxiliary_warmup_steps)
                loss = output["action_loss"] + cfg.loss_weight * ramp * (
                    components[0] + cfg.difference_loss_weight * components[1])
            finite = torch.isfinite(loss.detach()).all().int()
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite V3 objective")
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                skipped = self.accelerator.optimizer_step_was_skipped or bool(
                    getattr(getattr(self.model, "optimizer", None), "overflow", False))
                if skipped:
                    raise RuntimeError("Optimizer update skipped; refusing to advance clocks")
                if self.ema is not None:
                    self.ema.update(module)
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        names = diagnostic_names(cfg.trajectory_horizons)
        sums = torch.stack([output["action_loss"].detach().float(), output["position_sum"].detach().float(),
                            output["difference_sum"].detach().float(), output["position_count"].detach().float(),
                            output["difference_count"].detach().float()])
        diagnostics = output.get("point_diagnostics", torch.zeros(len(names), device=sums.device))
        sums = torch.cat([sums, diagnostics.detach().float()])
        self._step_sums = sums if self._step_sums is None else self._step_sums + sums
        if not self.accelerator.sync_gradients:
            return {}
        if self._pending_batches:
            raise RuntimeError("Accumulation boundary differs from prefetched window")
        if dist.is_initialized():
            dist.all_reduce(self._step_sums)
        action_sum, pos_sum, diff_sum = self._step_sums[:3]
        counts = self._step_sums[3:5]
        if not torch.equal(counts, self._global_counts):
            raise RuntimeError("Loss masks/counts differ from prefetched global window")
        pos, difference = torch.stack([pos_sum, diff_sum]) / counts.clamp_min(1)
        action = action_sum / (self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps)
        weighted = cfg.loss_weight * ramp * (pos + cfg.difference_loss_weight * difference)
        elapsed = time.perf_counter() - self._window_started
        metrics = dict(action_dit_loss=action.item(), position_loss=pos.item(), difference_loss=difference.item(),
                       total_loss=(action + weighted).item(), weighted_point_objective=weighted.item(),
                       point_track_weight=cfg.loss_weight * ramp, point_track_target_weight=cfg.loss_weight,
                       point_aux_ramp=ramp, position_valid_xyz=counts[0].item(), difference_valid_xyz=counts[1].item(),
                       point_track_valid_ratio=counts[0].item() / (self.total_batch_size * cfg.num_points * cfg.horizon * 3),
                       samples_per_second=self.total_batch_size / max(elapsed + self._window_data_time, 1e-6),
                       cuda_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        metrics.update({"timing/data": self._window_data_time, "timing/model": elapsed})
        if probe:
            metrics.update({name: value.item() for name, value in zip(names, self._step_sums[5:])})
        for name, p in module.policy_model.named_parameters():
            if name in before:
                metrics["point_update/" + name] = (p.detach().float() - before[name]).abs().max().item()
        if norm is None and hasattr(self.model, "get_global_grad_norm"):
            norm = self.model.get_global_grad_norm()
        if norm is not None:
            metrics["gradient_norm"] = float(norm)
        return metrics
