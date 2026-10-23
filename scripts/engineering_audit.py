"""Evidence-only hooks for <=10-step native GPU smoke; no optimizer changes."""
import json
from pathlib import Path
import torch


def audited_parameter_names(policy):
    """Include the actual Z_t source as well as the shared B/C point decoder."""
    from openpi.models_pytorch.position_point import validate_position_policy
    direct_query = validate_position_policy(policy)
    names = [
        "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.k_proj.weight",
        "point_track_head.point_mlp.0.weight", "point_track_head.point_mlp.2.weight",
        "point_track_head.fusion_in.weight", "point_track_head.fusion_out.weight"]
    if direct_query:
        names.append("future_queries")
    else:
        names.append("point_track_head.embedding.queries")
        for i in range(policy.point_track.embedding_layers):
            names += [f"point_track_head.embedding.layers.{i}.{tail}" for tail in
                      ("self_attention.in_proj_weight", "cross_attention.in_proj_weight", "ff.0.weight")]
    for i in range(policy.point_track.temporal_layers):
        names += [f"point_track_head.temporal.{i}.{tail}" for tail in
                  ("attention.in_proj_weight", "ff.0.weight")]
    return names


def install(trainer):
    original = trainer._train_step

    def audited_step(self, batch_vla, batch_vlm=None):
        from deepspeed.utils import safe_get_full_fp32_param
        if self.accelerator.gradient_accumulation_steps != 1 or self.total_batch_size != 64:
            raise ValueError("Engineering audit requires native global64/acc1")
        policy = self.accelerator.unwrap_model(self.model).policy_model
        params = dict(policy.named_parameters())
        names = audited_parameter_names(policy)
        before, gradients = {}, {}
        for name in names:
            master = safe_get_full_fp32_param(params[name])
            if master is None:
                raise RuntimeError("Missing native master: " + name)
            before[name] = master.detach().cpu().clone()
        handles = []
        for name in names:
            def capture(g, key=name):
                gradients[key] = float(g.detach().float().square().sum().cpu())
                return g
            handles.append(params[name].register_hook(capture))
        try:
            metrics = original(self, batch_vla, batch_vlm)
        finally:
            for handle in handles:
                handle.remove()
        proof = dict(step=self.completed_steps + 1, rank=self.accelerator.process_index,
                     world_size=self.accelerator.num_processes, global_batch=64, accumulation=1,
                     masters={})
        for name in names:
            master = safe_get_full_fp32_param(params[name]).detach().cpu()
            change = float((master - before[name]).abs().max())
            grad = gradients.get(name, 0.)
            if not torch.isfinite(master).all() or not (0 < grad < float("inf") and 0 < change < float("inf")):
                raise RuntimeError(f"Missing finite gradient/FP32 update: {name}: {grad}, {change}")
            proof["masters"][name] = dict(gradient_squared_norm=grad, update_max=change)
        output = Path(self.config.output_dir)
        with (output / f"position_native_audit_rank{proof['rank']}.jsonl").open("a") as stream:
            stream.write(json.dumps(proof) + "\n")
        if self.accelerator.is_main_process and not (output / "engineering_sample.pt").exists():
            torch.save(batch_vla[0], output / "engineering_sample.pt")
        print("POSITION_NATIVE_STEP_AUDIT " + json.dumps(proof), flush=True)
        return metrics

    trainer._train_step = audited_step
