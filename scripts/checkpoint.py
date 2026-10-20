"""Read-only same-world native state validation. Never loads untrusted checkpoints."""
from pathlib import Path
import json,re
from common import is_point_variant

def required_files(world,fixed=False):
    return ['latest','custom_checkpoint_0.pkl','custom_checkpoint_1.pkl',
        *(['custom_checkpoint_2.pkl'] if fixed else []),
        *[f'random_states_{r}.pkl' for r in range(world)],
        'pytorch_model/mp_rank_00_model_states.pt',
        *[f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(world)]]

def validate(path,cfg,deep=True):
    path=Path(path);step=int(re.fullmatch(r'checkpoint-([1-9][0-9]*)',path.name)[1])
    world=cfg['portable']['world_size'];fixed=cfg['portable']['variant']=='query_fixed'
    for name in required_files(world,fixed):
        f=path/name
        if f.is_symlink() or not f.is_file() or f.stat().st_size==0:raise ValueError(f'Incomplete checkpoint: {f}')
    if (path/'latest').read_text().strip()!='pytorch_model':raise ValueError('Wrong DeepSpeed tag')
    if len(list(path.glob('random_states_*.pkl')))!=world:raise ValueError('Wrong RNG world size')
    if len(list((path/'pytorch_model').glob('*_optim_states.pt')))!=world:raise ValueError('Wrong optimizer world size')
    weights=path.parent/f'steps_{step}_model.safetensors'
    if not weights.is_file() or not weights.stat().st_size:raise ValueError('Missing evaluation export')
    if not deep:return step
    import torch
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    progress=load(path/'custom_checkpoint_1.pkl');scheduler=load(path/'custom_checkpoint_0.pkl')
    if progress['completed_steps']!=step or progress['consumed_batches']!=step or scheduler['last_epoch']!=step:
        raise ValueError('Scheduler/progress clocks disagree')
    m=load(path/'pytorch_model/mp_rank_00_model_states.pt');ds=m['ds_config']
    if (m['dp_world_size'],m['mp_world_size'],m['global_steps'],m['global_samples'])!=(world,1,step,step*64):
        raise ValueError('DeepSpeed clocks/world differ')
    if (ds['train_micro_batch_size_per_gpu'],ds['train_batch_size'],ds['gradient_accumulation_steps'])!=(64//world,64,1):raise ValueError('Batch recipe differs')
    del m
    for rank in range(world):
        o=load(path/f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')['optimizer_state_dict']
        counts=o['partition_count'];counts=[counts] if isinstance(counts,int) else counts
        if any(n!=world for n in counts):raise ValueError('Partition count differs')
        states=o['base_optimizer_state']['state']
        if not states:raise ValueError('Empty Adam state')
        for s in states.values():
            if float(s['step'])!=step or s['exp_avg'].shape!=s['exp_avg_sq'].shape or s['exp_avg'].dtype!=torch.float32:raise ValueError('Adam clock/shape/dtype differs')
        del o,states
        rng=torch.load(path/f'random_states_{rank}.pkl',map_location='cpu',weights_only=False)
        if not all(k in rng for k in ('random_state','numpy_random_seed','torch_manual_seed','torch_cuda_manual_seed')):raise ValueError('RNG missing')
    if fixed:
        teacher=load(path/'custom_checkpoint_2.pkl')
        if int(teacher.get('updates',-1))!=step:raise ValueError('FIXED teacher clock differs')
    if is_point_variant(cfg['portable']['variant']):
        from starVLA.training.trajectory_contract import validate_contract
        validate_contract(path/'trajectory_contract.json',cfg,step)
    from safetensors import safe_open
    with safe_open(str(weights),framework='pt',device='cpu') as f:
        if not f.keys():raise ValueError('Empty model')
    return step

def choose_latest(run,cfg,validator=validate):
    candidates=[];rejected=[]
    for p in (Path(run)/'checkpoints').glob('checkpoint-*'):
        m=re.fullmatch(r'checkpoint-([1-9][0-9]*)',p.name)
        if m and p.is_dir():candidates.append((int(m[1]),p))
    for step,path in sorted(candidates,reverse=True):
        try:validator(path,cfg);return path,rejected
        except Exception as e:rejected.append({'step':step,'reason':str(e)})
    raise RuntimeError('No complete compatible checkpoint; refusing base restart: '+json.dumps(rejected))
