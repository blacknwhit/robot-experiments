"""Foresight reduction over the complete optimizer-step accumulation window."""
from collections import deque
from contextlib import nullcontext

import numpy as np
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist

from starVLA.training.train_starvla import VLATrainer, pi05_official_precision_enabled


class ForesightTrainer(VLATrainer):
    def _register_auxiliary_checkpoints(self):
        module = self.accelerator.unwrap_model(self.model)
        if module.policy_model.foresight.teacher_ema_decay is not None:
            teacher = module.get_future_teacher(self.accelerator.device)
            teacher.to(self.accelerator.device)
            # Scheduler=0, progress=1, teacher=2. Registration must precede load_state.
            self.accelerator.register_for_checkpointing(teacher)

    def _teacher_audit(self, phase="initialized"):
        module = self.accelerator.unwrap_model(self.model)
        if module.policy_model.foresight.teacher_ema_decay is None:
            return
        teacher = module.get_future_teacher(self.accelerator.device)
        assert teacher.updates == self.completed_steps, "Teacher and optimizer clocks differ"
        report = teacher.audit()
        if dist.is_initialized():
            reports = [None] * dist.get_world_size()
            dist.all_gather_object(reports, report)
            assert all(r == report for r in reports), "Teacher differs across data-parallel ranks"
        report.update(rank=self.accelerator.process_index, resumed=bool(self.resume_from_checkpoint),
                      cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9,
                      cuda_reserved_peak_gb=torch.cuda.max_memory_reserved()/1e9)
        (Path(self.config.output_dir) / f"teacher_audit_{phase}_rank{self.accelerator.process_index}.json").write_text(
            json.dumps(report, indent=2))


    def prepare_training(self):
        ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        self._point_track_enabled = self.model.policy_model.point_track.enabled
        required = {name: p for name, p in self.model.named_parameters()
                    if "future_queries" in name or "future_decoder" in name or
                    (self._point_track_enabled and "point_track_head" in name)}
        assert required and all(id(p) in ids for p in required.values())
        super().prepare_training()
        module = self.accelerator.unwrap_model(self.model)
        if self.resume_from_checkpoint:
            from starVLA.training.foresight_resume_audit import verify_loaded_partition
            verify_loaded_partition(self.model, self.resume_from_checkpoint, self.config.output_dir)
        teacher = module.get_future_teacher(self.accelerator.device)
        self._teacher_audit()
        if self.accelerator.is_main_process:
            runtime = {"gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "world_size": self.accelerator.num_processes,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
            (Path(self.config.output_dir) / "runtime_hardware.json").write_text(json.dumps(runtime, indent=2))
            self.accelerator.print("FORESIGHT_RUNTIME", runtime)
        self.accelerator.print("FORESIGHT_TEACHER", teacher.identity)
        self.accelerator.print("FORESIGHT_CONFIG", module.policy_model.foresight)
        bidir = module.policy_model.foresight.attention_mode == "bidirectional_xq"
        self.accelerator.print("FORESIGHT_MASK", [[1, int(bidir), 0], [1, 1, 0], [1, 1, 1]])

    def _get_next_batch(self):
        if not hasattr(self, "_pending_batches"):
            self._pending_batches = deque()
        if not self._pending_batches:
            start = time.perf_counter()
            g = self.accelerator.gradient_accumulation_steps
            parent_get = super()._get_next_batch
            self._pending_batches.extend(parent_get() for _ in range(g))
            if bool(self.config.trainer.get("acceptance_fault_masks", False)):
                if self.completed_steps == 1 or (self.completed_steps == 0 and self.accelerator.process_index == 0):
                    for batch in self._pending_batches:
                        for sample in batch:
                            sample["future_valid"] = np.zeros_like(sample["future_valid"])
            count = sum(int(np.asarray(sample["future_valid"]).sum())
                        for batch in self._pending_batches for sample in batch)
            self._future_global_count = torch.tensor(float(count), device=self.accelerator.device)
            if dist.is_initialized():
                dist.all_reduce(self._future_global_count)
            if self._point_track_enabled:
                point_count = 3 * sum(int(np.asarray(sample["point_valid"], dtype=bool).sum())
                                      for batch in self._pending_batches for sample in batch)
                self._point_global_count = torch.tensor(float(point_count), device=self.accelerator.device)
                if dist.is_initialized():
                    dist.all_reduce(self._point_global_count)
            self._step_sums = None
            self._window_data_time = time.perf_counter() - start
            self._window_model_time = 0.
        return self._pending_batches.popleft()

    def _train_step(self, batch_vla, batch_vlm=None):
        from openpi.models_pytorch.foresight import distributed_future_loss
        started = time.perf_counter()
        if self._window_model_time == 0:
            self._window_started = started
        query_before = None
        point_before = None
        with self.accelerator.accumulate(self.model):
            probe = self.completed_steps < 5 or (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0
            if self.accelerator.sync_gradients and probe:
                query_before = self.accelerator.unwrap_model(self.model).policy_model.future_queries.detach().float().clone()
                if self._point_track_enabled:
                    head = self.accelerator.unwrap_model(self.model).policy_model.point_track_head
                    point_before = {k: p.detach().float().clone() for k, p in head.named_parameters()
                                    if k in ('time_queries', 'fusion_out.weight', 'point_mlp.0.weight')}
            context = nullcontext() if pi05_official_precision_enabled(self.config) else torch.autocast("cuda", dtype=torch.bfloat16)
            with context:
                output = self.model(batch_vla)
                cfg = self.accelerator.unwrap_model(self.model).policy_model.foresight
                weight = cfg.weight(self.completed_steps)
                future = distributed_future_loss(output["future_sum"], self._future_global_count,
                    world_size=self.accelerator.num_processes,
                    accumulation_steps=self.accelerator.gradient_accumulation_steps)
                loss = output["action_loss"] + weight * future
                if self._point_track_enabled:
                    from openpi.models_pytorch.point_track import distributed_point_track_loss
                    point_cfg = self.accelerator.unwrap_model(self.model).policy_model.point_track
                    point_loss = distributed_point_track_loss(output["point_sum"], self._point_global_count,
                        world_size=self.accelerator.num_processes,
                        accumulation_steps=self.accelerator.gradient_accumulation_steps)
                    loss = loss + point_cfg.loss_weight * point_loss
            finite = torch.isfinite(loss.detach()).all().int()
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f"Nonfinite foresight loss at update {self.completed_steps}")
            self.accelerator.backward(loss)
            norm = None
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                if cfg.teacher_ema_decay is not None:
                    skipped = self.accelerator.optimizer_step_was_skipped
                    skipped = skipped or bool(getattr(getattr(self.model, "optimizer", None), "overflow", False))
                    if skipped:
                        raise RuntimeError("Optimizer step skipped; do not advance teacher or progress")
                    module = self.accelerator.unwrap_model(self.model)
                    vision = module.policy_model.paligemma_with_expert.paligemma.model.vision_tower
                    module.get_future_teacher(self.accelerator.device).update(vision, self.completed_steps + 1)
                if self.ema is not None:
                    self.ema.update(self.accelerator.unwrap_model(self.model))
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        sums = torch.cat((output["future_head_sum"].detach().flatten(),
                          output["future_head_count"].detach().flatten().float(),
                          output["action_loss"].detach().reshape(1))).float()
        if self._point_track_enabled:
            sums = torch.cat((sums, output['point_sum'].detach().reshape(1).float(),
                              output['point_count'].detach().reshape(1).float()))
        self._step_sums = sums if self._step_sums is None else self._step_sums + sums
        self._window_model_time += time.perf_counter() - started
        if not self.accelerator.sync_gradients:
            return {}
        if self._pending_batches:
            raise RuntimeError("Accelerate synchronization does not match the prefetched accumulation window")
        if dist.is_initialized():
            dist.all_reduce(self._step_sums)
        heads = len(cfg.future_offsets) * len(cfg.future_view_keys)
        numerator, counts = self._step_sums[:heads], self._step_sums[heads:2 * heads]
        mean = numerator.sum() / counts.sum().clamp_min(1)
        action = self._step_sums[2 * heads] / (self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps)
        metrics = {"action_dit_loss": action.item(), "future_loss": mean.item(),
                   "total_loss": (action + weight * mean).item(), "future_weight": weight,
                   "future_valid_count": counts.sum().item(),
                   "future_valid_ratio": counts.sum().item() / (self.total_batch_size * heads),
                   "cuda_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        if self._point_track_enabled:
            point_sum, point_count = self._step_sums[-2:]
            if not torch.equal(point_count, self._point_global_count):
                raise RuntimeError('Point loss mask/count differs from prefetched optimization window')
            point_mean = point_sum / point_count.clamp_min(1)
            metrics.update(point_track_loss=point_mean.item(), point_track_valid_xyz=point_count.item(),
                           point_track_weight=point_cfg.loss_weight,
                           point_track_valid_ratio=point_count.item() / (self.total_batch_size *
                               point_cfg.num_points * point_cfg.horizon * 3),
                           total_loss=(action + weight * mean + point_cfg.loss_weight * point_mean).item())
            if point_before is not None:
                head = self.accelerator.unwrap_model(self.model).policy_model.point_track_head
                for name, p in head.named_parameters():
                    if name in point_before:
                        metrics['point_update/' + name] = (p.detach().float() - point_before[name]).abs().max().item()
        if cfg.teacher_ema_decay is not None:
            teacher = self.accelerator.unwrap_model(self.model).get_future_teacher(self.accelerator.device)
            metrics.update(teacher_ema_decay=teacher.decay, teacher_updates=teacher.updates)
        for i, value in enumerate(numerator / counts.clamp_min(1)):
            metrics[f"future/head_{i}"] = value.item()
        if norm is None and hasattr(self.model, "get_global_grad_norm"):
            norm = self.model.get_global_grad_norm()
        if norm is not None:
            metrics["gradient_norm"] = float(norm)
        if query_before is not None:
            query = self.accelerator.unwrap_model(self.model).policy_model.future_queries.detach().float()
            metrics["query_update_max"] = (query - query_before).abs().max().item()
            metrics["query_weight_norm"] = query.norm().item()
        self._window_model_time = time.perf_counter() - self._window_started
        metrics["timing/data"] = self._window_data_time
        metrics["timing/model"] = self._window_model_time
        metrics["samples_per_second"] = self.total_batch_size / max(self._window_data_time + self._window_model_time, 1e-6)
        return metrics

    def _log_metrics(self, metrics):
        super()._log_metrics(metrics)
        if self.accelerator.is_main_process and (self.completed_steps <= 5 or
                self.completed_steps % self.config.trainer.logging_frequency == 0):
            with (Path(self.config.output_dir) / "foresight_metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step":self.completed_steps, **metrics}) + "\n")


    def _save_checkpoint(self):
        self._teacher_audit(phase=f"step{self.completed_steps}")
        super()._save_checkpoint()
        control = self.config.trainer.get("auto_resume_control_dir")
        if not control:
            return
        error = [None]
        if self.accelerator.is_main_process:
            try:
                from starVLA.training.foresight_recovery import commit_and_prune
                commit_and_prune(self.config.output_dir, self.completed_steps, control)
            except Exception as exc:
                error[0] = f"Rolling checkpoint commit failed: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])

    def _finalize_training(self):
        if not self.config.trainer.get("auto_resume_control_dir"):
            return super()._finalize_training()
        goal = self.config.trainer.max_train_steps
        stop = min(goal, int(self.config.trainer.get("stop_after_steps", goal)))
        if self.completed_steps != stop:
            raise RuntimeError("Managed training ended before the target update count")
        if stop < goal:
            self._teacher_audit(phase="probe_final")
        # The final full state and evaluation weights were committed by _save_checkpoint.
        # Avoid writing a second copy under final_model; retention owns the saved results.
        self.accelerator.wait_for_everyone()
        marker = "FORESIGHT_MANAGED_PROBE_COMPLETE" if stop < goal else "FORESIGHT_MANAGED_TRAINING_COMPLETE"
        self.accelerator.print(marker, self.completed_steps)
