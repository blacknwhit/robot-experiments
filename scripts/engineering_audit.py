"""Evidence-only hooks for <=10-step native GPU smoke; no optimizer changes."""
import json
from pathlib import Path
import torch


def audited_parameter_names(policy):
    """Include the actual Z_t source as well as the shared B/C point decoder."""
    paper = not hasattr(policy.point_track, 'objective_version') and policy.point_track.version == 'pri4r_embedding_v1'
    if paper:
        if policy.foresight.enabled:raise ValueError('Pri4R must preserve original prefix')
        direct_query = False
    else:
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
    for i in range(getattr(policy.point_track,'temporal_layers',0)):
        names += [f"point_track_head.temporal.{i}.{tail}" for tail in
                  ("attention.in_proj_weight", "ff.0.weight")]
    return names


def install(trainer):
    original = trainer._train_step

    def audited_step(self, batch_vla, batch_vlm=None):
        if self.config.framework.pi05.get('official_precision_mode',False):
            return native_audited_step(original,self,batch_vla,batch_vlm)
        from deepspeed.utils import safe_get_full_fp32_param
        accumulation = self.accelerator.gradient_accumulation_steps
        if self.total_batch_size not in (64,128):
            raise ValueError("Engineering audit requires native global64/128")
        policy = self.accelerator.unwrap_model(self.model).policy_model
        params = dict(policy.named_parameters())
        names = audited_parameter_names(policy)
        if not hasattr(self,'_native_audit_window'):
            before={}
            for name in names:
                master = safe_get_full_fp32_param(params[name])
                if master is None:
                    raise RuntimeError("Missing native master: " + name)
                before[name] = master.detach().cpu().clone()
            self._native_audit_window={'before':before,'gradients':{},'microbatches':0}
        window=self._native_audit_window
        before,gradients=window['before'],window['gradients']
        handles = []
        for name in names:
            def capture(g, key=name):
                gradients[key] = gradients.get(key,0.) + float(g.detach().float().square().sum().cpu())
                return g
            handles.append(params[name].register_hook(capture))
        try:
            metrics = original(self, batch_vla, batch_vlm)
        finally:
            for handle in handles:
                handle.remove()
        window['microbatches']+=1
        if not self.accelerator.sync_gradients:
            return metrics
        if window['microbatches']!=accumulation:
            raise RuntimeError('Audit accumulation window differs from optimizer boundary')
        proof = dict(step=self.completed_steps + 1, rank=self.accelerator.process_index,
                     world_size=self.accelerator.num_processes, global_batch=self.total_batch_size, accumulation=accumulation,
                     masters={})
        zero=getattr(self.model,'optimizer',None)
        proof['optimizer_class']=type(getattr(zero,'optimizer',None)).__name__
        proof['cpu_offload']=bool(getattr(zero,'cpu_offload',False))
        proof['ema_device']=str(next(iter(self.ema.shadow_parameters.values())).device) if self.ema else None
        if self.config.trainer.get('optimizer_offload_cpu',False):
            if not proof['cpu_offload'] or proof['optimizer_class']!='AdamW' or proof['ema_device']!='cpu':
                raise RuntimeError('Actual optimizer/EMA placement differs from single-GPU contract')
        for name in names:
            actual_master=safe_get_full_fp32_param(params[name])
            if actual_master.dtype != torch.float32:raise RuntimeError('Master is not FP32: '+name)
            master = actual_master.detach().cpu()
            change = float((master - before[name]).abs().max())
            grad = gradients.get(name, 0.)
            if not torch.isfinite(master).all() or not (0 < grad < float("inf") and 0 < change < float("inf")):
                raise RuntimeError(f"Missing finite gradient/FP32 update: {name}: {grad}, {change}")
            proof["masters"][name] = dict(gradient_squared_norm=grad, update_max=change, actual_device=str(actual_master.device))
        output = Path(self.config.output_dir)
        with (output / f"position_native_audit_rank{proof['rank']}.jsonl").open("a") as stream:
            stream.write(json.dumps(proof) + "\n")
        if self.accelerator.is_main_process and not (output / "engineering_sample.pt").exists():
            torch.save(batch_vla[0], output / "engineering_sample.pt")
        print("POSITION_NATIVE_STEP_AUDIT " + json.dumps(proof), flush=True)
        del self._native_audit_window
        return metrics

    trainer._train_step = audited_step


def native_audited_step(original,trainer,batch_vla,batch_vlm=None):
    """Inspect actual DDP parameter updates; BF16 has no hidden FP32 master."""
    policy=trainer.accelerator.unwrap_model(trainer.model).policy_model
    params=dict(policy.named_parameters());names=audited_parameter_names(policy)
    if not hasattr(trainer,'_native_audit_window'):
        trainer._native_audit_window={'before':{n:params[n].detach().cpu().clone() for n in names},
                                      'gradients':{},'microbatches':0,'local_valid_xyz':0}
    window=trainer._native_audit_window;handles=[]
    window['local_valid_xyz'] += 3 * sum(int(s['point_valid'].sum()) for s in batch_vla)
    for name in names:
        def capture(g,key=name):
            window['gradients'][key]=window['gradients'].get(key,0.)+float(g.detach().float().square().sum().cpu())
            return g
        handles.append(params[name].register_hook(capture))
    try:metrics=original(trainer,batch_vla,batch_vlm)
    finally:
        for handle in handles:handle.remove()
    window['microbatches']+=1
    if not trainer.accelerator.sync_gradients:return metrics
    if window['microbatches']!=trainer.accelerator.gradient_accumulation_steps:
        raise ValueError('Native audit accumulation mismatch')
    proof={'step':trainer.completed_steps+1,'rank':trainer.accelerator.process_index,
           'world_size':trainer.accelerator.num_processes,'global_batch':trainer.total_batch_size,
           'backend':'native_ddp','has_fp32_master_copy':False,'parameters':{}}
    proof.update(local_valid_xyz=window['local_valid_xyz'],
                 global_valid_xyz=int(trainer._point_global_count.item()))
    for name in names:
        p=params[name];now=p.detach().cpu();grad=window['gradients'].get(name,0.)
        change=float((now.float()-window['before'][name].float()).abs().max())
        auxiliary=name.startswith('point_track_head.')
        expected_local_gradient=not auxiliary or proof['local_valid_xyz']>0
        expected_global_update=not auxiliary or proof['global_valid_xyz']>0
        if not torch.isfinite(now).all() or not 0<=grad<float('inf') or not 0<=change<float('inf'):
            raise RuntimeError('Missing finite parameter/gradient: '+name)
        if expected_local_gradient and grad == 0:
            raise RuntimeError('Missing gradient with valid supervision: '+name)
        if auxiliary and not expected_local_gradient and grad != 0:
            raise RuntimeError('Auxiliary gradient leaked from an all-invalid local window: '+name)
        # The warmup update can round to zero in upstream-native BF16 parameters.
        # Report this explicitly; never pretend a ZeRO master was updated.
        if p.dtype==torch.float32 and expected_global_update and change == 0:
            raise RuntimeError('No FP32 auxiliary update: '+name)
        proof['parameters'][name]={'dtype':str(p.dtype),'gradient_squared_norm':grad,'update_max':change,
                                  'local_gradient_required':expected_local_gradient,
                                  'global_update_required':expected_global_update}
    output=Path(trainer.config.output_dir)
    with (output/f"native_update_rank{proof['rank']}.jsonl").open('a') as f:f.write(json.dumps(proof)+'\n')
    if trainer.accelerator.is_main_process and not (output/'engineering_sample.pt').exists():
        torch.save(batch_vla[0],output/'engineering_sample.pt')
    print('NATIVE_DDP_STEP_AUDIT '+json.dumps(proof),flush=True)
    del trainer._native_audit_window
    return metrics
