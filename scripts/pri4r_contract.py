"""Disclosed Pri4R recipe and bounded-data contract. Imports no GPU libraries."""
import json
from pathlib import Path
from common import sha256


def validate_recipe(c):
    t=c['trainer'];p=c['framework']['pi05'];d=c['datasets']['vla_data'];v=c['portable']
    expected={'max_train_steps':30000,'num_warmup_steps':10000,'ema_decay':0.999,
              'expected_global_batch_size':128,'ema_missing_resume_policy':'error'}
    if any(t.get(k)!=value for k,value in expected.items()):
        raise ValueError('Pri4R paper training recipe changed')
    if t['learning_rate']['base']!=5e-5 or t['lr_scheduler_type']!='openpi_cosine':
        raise ValueError('Pri4R LR/schedule differs')
    pt=p['point_track']
    if not p.get('official_precision_mode') or t.get('optimizer_offload_cpu'):
        raise ValueError('Pri4R alignment requires native OpenPI precision/AdamW')
    if pt['coordinate_frame']!='world' or d.get('point_track_coordinate_frame')!='world':
        raise ValueError('User selected world-coordinate point inputs and targets')
    if pt.get('clock_policy') != 'physical_window_alignment_v1' or d.get('point_track_clock_policy') != pt['clock_policy']:
        raise ValueError('Pri4R requires recorded physical-clock auxiliary supervision')
    if (pt['version'],pt['num_points'],pt['loss_weight'],pt['output_kind'])!=('pri4r_embedding_v1',1024,1.0,'adjacent_displacement'):
        raise ValueError('Pri4R auxiliary method differs')
    if p['foresight']['enabled'] or p['foresight']['auxiliary_enabled'] or d['group_sampling']:
        raise ValueError('Extra Query/teacher/G8 method is not Pri4R paper mode')
    smoke=bool(v.get('smoke'));formal=bool(v.get('formal'))
    if smoke==formal or not 1<=t['stop_after_steps']<=(10 if smoke else 30000):
        raise ValueError('Choose explicit bounded smoke or formal training')
    if bool(d.get('pri4r_smoke_manifest'))!=smoke:
        raise ValueError('Partial data allowed only in explicit bounded smoke')
    world=v['world_size'];acc=t['gradient_accumulation_steps'];micro=d['per_device_batch_size']
    if world*acc*micro!=128:raise ValueError('Pri4R requires global batch128, including single-GPU smoke')
    if world==1:
        if not smoke or (micro,acc)!=(4,32) or t.get('ema_device')!='cpu':
            raise ValueError('Single GPU requires bounded native AdamW with CPU EMA')
    elif t.get('optimizer_offload_cpu') or t.get('ema_device','cuda')!='cuda':
        raise ValueError('Formal multi-GPU recipe changed its state placement')


def validate_gt_receipt(c,proof,gt):
    gt=Path(gt);d=c['datasets']['vla_data']
    if proof.get('method')!='pri4r_paper' or not proof.get('verified_all_array_hashes'):
        raise ValueError('GT receipt is not an audited paper-sampling dataset')
    if proof.get('clock_policy') != 'physical_window_alignment_v1':
        raise ValueError('Old GT receipt lacks physical-clock verification; regenerate and verify new GT')
    if proof['config_sha256']!=d['point_track_gt_config_sha256']:
        raise ValueError('GT sampling configuration mismatch')
    if proof['manifest_sha256']!=sha256(gt/'input_manifest.json'):
        raise ValueError('GT input manifest changed')
    if c['portable'].get('smoke'):
        if not proof.get('bounded') or not 1<=proof['episodes']<=8:
            raise ValueError('Smoke requires an explicitly bounded 1..8 episode receipt')
        if Path(d['pri4r_smoke_manifest']).resolve()!=(gt/'input_manifest.json').resolve():
            raise ValueError('Smoke dataloader and verified GT manifest differ')
    elif proof.get('bounded') or (proof['episodes'],proof['frames'])!=(30000,27815381):
        raise ValueError('Formal RoboCasa365 training requires the entire verified scope')


def validate_ema(path,step,config):
    import torch
    from safetensors import safe_open
    path=Path(path)
    ema=path/'ema_model.safetensors'
    raw=path.parent/f'steps_{step}_raw_model.safetensors'
    export=path.parent/f'steps_{step}_model.safetensors'
    if not all(p.is_file() and p.stat().st_size for p in (ema,raw,export)):
        raise ValueError('Missing EMA state/raw comparison/export')
    with safe_open(str(ema),framework='pt',device='cpu') as e, safe_open(str(raw),framework='pt',device='cpu') as r:
        m=e.metadata() or {}
        if (float(m.get('ema_decay',-1)),int(m.get('ema_num_updates',-1)),m.get('weight_role'))!=(0.999,step,'ema_inference'):
            raise ValueError('EMA update clock/decay/role mismatch')
        if set(e.keys())!=set(r.keys()):raise ValueError('EMA/raw tensor topology mismatch')
        aliases=json.loads(m.get('ema_aliases','{}'))
        if not aliases or not (set(aliases)|set(aliases.values())) <= set(e.keys()):
            raise ValueError('Missing/invalid EMA parameter alias topology')
        for key in r.keys():
            a,b=e.get_slice(key),r.get_slice(key)
            if a.get_shape()!=b.get_shape():raise ValueError('EMA tensor shape mismatch: '+key)
            if b.get_dtype() in ('F32','BF16','F16') and a.get_dtype()!='F32':
                raise ValueError('EMA floating tensor is not FP32: '+key)
            value=e.get_tensor(key)
            if value.is_floating_point():
                flat=value.flatten()
                for start in range(0,flat.numel(),1048576):
                    if not torch.isfinite(flat[start:start+1048576]).all():
                        raise ValueError('Nonfinite EMA state: '+key)
            primary=aliases.get(key,key)
            if primary!=key and not torch.equal(value,e.get_tensor(primary)):
                raise ValueError('EMA tied aliases differ: '+key)
    # Export is an immutable hardlink/copy of the same EMA checkpoint.
    if not ema.samefile(export) and sha256(ema)!=sha256(export):
        raise ValueError('Evaluation export differs from saved EMA')
