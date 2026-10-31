"""Action + direct-Query point loss; no future-image data, decoder or teacher.

The established native model/optimizer/scheduler/RNG checkpoints are inherited.
No teacher or legacy Query64 retention code is used. Reductions cover the whole
optimizer accumulation window, including unequal EOF validity across ranks.
"""
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.distributed as dist

from starVLA.training.train_starvla import VLATrainer, pi05_official_precision_enabled


class QueryPointTrainer(VLATrainer):
    def _save_checkpoint(self):
        VLATrainer._save_checkpoint(self)
        if self.config.trainer.get('rolling_retention', False):
            if self.accelerator.is_main_process:
                from starVLA.training.query_point_retention import complete_and_prune
                complete_and_prune(self.config.output_dir, self.completed_steps,
                                   self.accelerator.num_processes,
                                   self.accelerator.gradient_accumulation_steps)
            self.accelerator.wait_for_everyone()

    def _register_auxiliary_checkpoints(self):
        # Head and Query parameters/signatures are in the normal model state.
        # Scheduler, optimizer, RNG and progress use the native base trainer.
        pass

    def prepare_training(self):
        p = self.model.policy_model
        embedding = p.point_track.enabled and p.point_track.version == "pri4r_embedding_v1"
        if embedding:
            if p.foresight.enabled or hasattr(p, "future_queries") or p.foresight.auxiliary_enabled:
                raise ValueError("Embedding variant must retain the original pi0.5 prefix")
        elif (p.foresight.version != "prefix_query_track_v1" or not p.point_track.enabled
                or p.point_track.version != "pri4r_query_zt_v1" or p.foresight.auxiliary_enabled
                or getattr(self.model, "_teacher", None) is not None):
            raise ValueError("QueryPointTrainer requires the direct-Query, point-only contract")
        if getattr(self.model, '_teacher', None) is not None:
            raise ValueError('No future-image teacher allowed')
        ids = {id(x) for group in self.optimizer.param_groups for x in group['params']}
        required = {name: x for name, x in self.model.named_parameters()
                    if 'future_queries' in name or 'point_track_head' in name}
        if not required or not all(x.requires_grad and id(x) in ids for x in required.values()):
            raise ValueError("Query and point-head parameters must be trainable and present in optimizer")
        VLATrainer.prepare_training(self)
        if self.ema is None or self.ema.decay != 0.999 or self.ema.num_updates != self.completed_steps:
            raise ValueError('Pri4R EMA must restore the exact update clock with decay0.999')
        if self.resume_from_checkpoint:
            from safetensors import safe_open
            ema_path = Path(self.resume_from_checkpoint)/'ema_model.safetensors'
            with safe_open(str(ema_path),framework='pt',device='cpu') as f:
                for name,shadow in self.ema.shadow_parameters.items():
                    if not torch.equal(shadow, f.get_tensor(name).to(shadow.device)):
                        raise ValueError('EMA resume audit differs: '+name)
            (Path(self.config.output_dir)/f'ema_restore_rank{self.accelerator.process_index}.json').write_text(
                json.dumps({'step':self.completed_steps,'decay':self.ema.decay,
                            'unique_parameters':len(self.ema.shadow_parameters),'bitwise_equal':True}))
        if self.resume_from_checkpoint:
            from starVLA.training.foresight_resume_audit import verify_loaded_partition
            verify_loaded_partition(self.model, self.resume_from_checkpoint, self.config.output_dir)
        if self.accelerator.is_main_process:
            runtime = {'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       'torch': str(torch.__version__), 'world_size': self.accelerator.num_processes,
                       'global_batch': self.total_batch_size, 'teacher': False, 'future_image_loss': False,
                       'query_shape': None if embedding else list(p.future_queries.shape),
                       'embedding_query_shape': list(p.point_track_head.embedding.queries.shape) if embedding else None,
                       'point_context': 'original_vlm_embedding_Zt' if embedding else 'query_output_direct_Zt'}
            (Path(self.config.output_dir)/'runtime_hardware.json').write_text(json.dumps(runtime, indent=2))
            self.accelerator.print('QUERY_POINT_RUNTIME', runtime)

    def _get_next_batch(self):
        if not hasattr(self, '_pending_batches'):
            self._pending_batches = deque()
        if not self._pending_batches:
            start = time.perf_counter()
            # Call the native loader directly, not the future-image prefetcher.
            for _ in range(self.accelerator.gradient_accumulation_steps):
                self._pending_batches.append(VLATrainer._get_next_batch(self))
            count = 3 * sum(int(np.asarray(s['point_valid'], dtype=bool).sum())
                            for batch in self._pending_batches for s in batch)
            self._point_global_count = torch.tensor(float(count), device=self.accelerator.device)
            if dist.is_initialized():
                dist.all_reduce(self._point_global_count)
            self._step_sums = None
            self._window_data_time = time.perf_counter() - start
            self._window_started = None
        return self._pending_batches.popleft()

    def _train_step(self, batch_vla, batch_vlm=None):
        from openpi.models_pytorch.point_track import distributed_point_track_loss
        if self._window_started is None:
            self._window_started = time.perf_counter()
        before = None
        norm = None
        with self.accelerator.accumulate(self.model):
            module = self.accelerator.unwrap_model(self.model)
            cfg = module.policy_model.point_track
            probe = self.completed_steps < 5 or (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0
            if self.accelerator.sync_gradients and probe:
                before = {name: p.detach().float().clone() for name, p in module.policy_model.named_parameters()
                          if name in ('future_queries', 'point_track_head.embedding.queries',
                                      'point_track_head.embedding.layers.0.self_attention.in_proj_weight',
                                      'point_track_head.embedding.layers.0.cross_attention.in_proj_weight',
                                      'point_track_head.point_mlp.0.weight',
                                      'point_track_head.fusion_in.weight', 'point_track_head.fusion_out.weight')}
            context = nullcontext() if pi05_official_precision_enabled(self.config) else torch.autocast('cuda', dtype=torch.bfloat16)
            with context:
                output = self.model(batch_vla)
                point_loss = distributed_point_track_loss(output['point_sum'], self._point_global_count,
                    world_size=self.accelerator.num_processes,
                    accumulation_steps=self.accelerator.gradient_accumulation_steps)
                loss = output['action_loss'] + cfg.loss_weight * point_loss
            finite = torch.isfinite(loss.detach()).all().int()
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite query-point loss at update {self.completed_steps}')
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                skipped = self.accelerator.optimizer_step_was_skipped or bool(
                    getattr(getattr(self.model, 'optimizer', None), 'overflow', False))
                if skipped:
                    raise RuntimeError('Optimizer step skipped; do not advance scheduler/progress')
                if self.ema is not None:
                    self.ema.update(module)
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        sums = torch.stack([output['action_loss'].detach().float(), output['point_sum'].detach().float(),
                            output['point_count'].detach().float()])
        self._step_sums = sums if self._step_sums is None else self._step_sums + sums
        if not self.accelerator.sync_gradients:
            return {}
        if self._pending_batches:
            raise RuntimeError('Gradient accumulation boundary differs from prefetched window')
        if dist.is_initialized():
            dist.all_reduce(self._step_sums)
        action_sum, point_sum, point_count = self._step_sums
        if not torch.equal(point_count, self._point_global_count):
            raise RuntimeError('Actual point mask/count differs from prefetched window')
        action = action_sum / (self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps)
        mean = point_sum / point_count.clamp_min(1)
        elapsed = time.perf_counter() - self._window_started
        metrics = {'action_dit_loss': action.item(), 'point_track_loss': mean.item(),
                   'total_loss': (action + cfg.loss_weight * mean).item(), 'point_track_weight': cfg.loss_weight,
                   'point_track_valid_xyz': point_count.item(), 'point_track_valid_ratio': point_count.item() /
                       (self.total_batch_size * cfg.num_points * cfg.horizon * 3),
                   'timing/data': self._window_data_time, 'timing/model': elapsed,
                   'samples_per_second': self.total_batch_size / max(elapsed + self._window_data_time, 1e-6),
                   'cuda_peak_gb': torch.cuda.max_memory_allocated()/1e9}
        if before is not None:
            for name, p in module.policy_model.named_parameters():
                if name in before:
                    value = (p.detach().float() - before[name]).abs().max().item()
                    metrics['query_update_max' if name == 'future_queries' else 'point_update/'+name] = value
        if norm is None and hasattr(self.model, 'get_global_grad_norm'):
            norm = self.model.get_global_grad_norm()
        if norm is not None:
            metrics['gradient_norm'] = float(norm)
        return metrics

    def _log_metrics(self, metrics):
        VLATrainer._log_metrics(self, metrics)
        if self.accelerator.is_main_process and (self.completed_steps <= 5 or
                self.completed_steps % self.config.trainer.logging_frequency == 0):
            with (Path(self.config.output_dir)/'point_track_metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': self.completed_steps, **metrics})+'\n')
