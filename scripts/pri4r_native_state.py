"""Native OpenPI AdamW state evidence. No fictitious ZeRO FP32 master weights."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def native_mode(cfg):
    return bool(cfg['framework']['pi05'].get('official_precision_mode', False))


def accumulation_plugin(steps):
    from accelerate.utils import GradientAccumulationPlugin
    # The trainer prefetches a complete optimizer window. An epoch boundary
    # observed during prefetch must not prematurely step the optimizer.
    return GradientAccumulationPlugin(num_steps=steps,sync_with_dataloader=False)


def tensor_equal(actual, expected, label):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError('Restored dtype/shape differs: '+label)
    a,b=actual.detach().reshape(-1),expected.detach().reshape(-1)
    for start in range(0,a.numel(),1048576):
        if not torch.equal(a[start:start+1048576].cpu(),b[start:start+1048576].cpu()):
            raise ValueError('Restored values differ: '+label)


def equal_tree(actual, expected, label):
    if isinstance(expected,torch.Tensor):
        tensor_equal(actual,expected,label)
    elif isinstance(expected,np.ndarray):
        if not np.array_equal(actual,expected):raise ValueError('Restored array differs: '+label)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError('Restored keys differ: '+label)
        for key in expected:equal_tree(actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,(tuple,list)):
        if len(actual)!=len(expected):raise ValueError('Restored length differs: '+label)
        for i,(a,b) in enumerate(zip(actual,expected)):equal_tree(a,b,label+'/'+str(i))
    elif actual!=expected:
        raise ValueError('Restored value differs: '+label)


def write_metadata(trainer,path):
    model=trainer.accelerator.unwrap_model(trainer.model)
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p),[]).append(name)
    groups=[]
    for group in trainer.optimizer.param_groups:
        groups.append([{'names':aliases[id(p)],'shape':list(p.shape),'dtype':str(p.dtype)}
                       for p in group['params']])
    report={'backend':'native_ddp','world':trainer.accelerator.num_processes,
            'global_batch':trainer.total_batch_size,
            'accumulation':trainer.accelerator.gradient_accumulation_steps,
            'step':trainer.completed_steps,'parameter_groups':groups,
            'active_parameter_ids':list(trainer.optimizer.state_dict()['state']),
            'has_fp32_master_copy':False,'optimizer':'torch.optim.AdamW'}
    (Path(path)/'native_state.json').write_text(json.dumps(report,indent=2))


def validate_native(path,cfg,step,deep=True):
    from safetensors import safe_open
    path=Path(path);world=cfg['portable']['world_size']
    required=['model.safetensors','optimizer.bin','native_state.json',
              'custom_checkpoint_0.pkl','custom_checkpoint_1.pkl',
              *[f'random_states_{r}.pkl' for r in range(world)]]
    for name in required:
        p=path/name
        if p.is_symlink() or not p.is_file() or not p.stat().st_size:
            raise ValueError('Incomplete native state: '+str(p))
    if (path/'pytorch_model').exists():raise ValueError('Mixed native/ZeRO checkpoint')
    meta=json.loads((path/'native_state.json').read_text())
    if (meta['backend'],meta['world'],meta['step'],meta['global_batch'],meta['accumulation']) != (
            'native_ddp',world,step,128,cfg['trainer']['gradient_accumulation_steps']):
        raise ValueError('Native checkpoint topology/clocks differ')
    if len(list(path.glob('random_states_*.pkl')))!=world:raise ValueError('Native RNG world differs')
    if not deep:return
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    progress=load(path/'custom_checkpoint_1.pkl');scheduler=load(path/'custom_checkpoint_0.pkl')
    if (progress['completed_steps'],progress['consumed_batches'],scheduler['last_epoch']) != (
            step,step*meta['accumulation'],step):raise ValueError('Native progress clocks disagree')
    optimizer=load(path/'optimizer.bin')
    if set(optimizer['state'])!=set(meta['active_parameter_ids']):
        raise ValueError('Missing native Adam parameter states')
    if len(optimizer['param_groups'])!=1 or len(meta['parameter_groups'])!=1:
        raise ValueError('Official AdamW requires one parameter group')
    group=optimizer['param_groups'][0];parameters=meta['parameter_groups'][0]
    if len(group['params'])!=len(parameters) or not optimizer['state']:
        raise ValueError('Native optimizer topology is incomplete')
    spec=cfg['trainer']['optimizer']
    if (tuple(group['betas']),group['eps'],group['weight_decay']) != (
            tuple(spec['betas']),spec['eps'],spec['weight_decay']):raise ValueError('AdamW recipe differs')
    with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as f:
        keys=set(f.keys())
        for pid,param in zip(group['params'],parameters):
            candidates=keys.intersection(param['names'])
            if not candidates:raise ValueError('Native weights missing: '+param['names'][0])
            weight=f.get_tensor(sorted(candidates)[0])
            if list(weight.shape)!=param['shape'] or str(weight.dtype)!=param['dtype']:
                raise ValueError('Native parameter dtype/shape mismatch')
            state=optimizer['state'].get(pid)
            # Native AdamW only creates states for parameters that received gradients.
            if state is None:continue
            if float(state['step'])!=step:raise ValueError('Native Adam clock mismatch')
            for key in ('exp_avg','exp_avg_sq'):
                value=state[key]
                if value.shape!=weight.shape or value.dtype!=weight.dtype:
                    raise ValueError('Native Adam dtype/shape mismatch: '+param['names'][0])
                for part in value.reshape(-1).split(1048576):
                    if not torch.isfinite(part).all():raise ValueError('Nonfinite Adam state')
    for rank in range(world):
        rng=load(path/f'random_states_{rank}.pkl')
        if not {'random_state','numpy_random_seed','torch_manual_seed','torch_cuda_manual_seed'}<=rng.keys():
            raise ValueError('Incomplete native RNG state')


def audit_restored(trainer):
    from safetensors import safe_open
    path=Path(trainer.resume_from_checkpoint);step=trainer.completed_steps
    model=trainer.accelerator.unwrap_model(trainer.model)
    actual=model.state_dict()
    with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as f:
        for key in f.keys():tensor_equal(actual[key],f.get_tensor(key),'model/'+key)
    saved=torch.load(path/'optimizer.bin',map_location='cpu',weights_only=False,mmap=True)
    equal_tree(trainer.optimizer.state_dict(),saved,'AdamW')
    rank=trainer.accelerator.process_index
    rng=torch.load(path/f'random_states_{rank}.pkl',map_location='cpu',weights_only=False)
    equal_tree(random.getstate(),rng['random_state'],'python_rng')
    equal_tree(np.random.get_state(),rng['numpy_random_seed'],'numpy_rng')
    tensor_equal(torch.get_rng_state(),rng['torch_manual_seed'],'torch_rng')
    equal_tree(torch.cuda.get_rng_state_all(),rng['torch_cuda_manual_seed'],'cuda_rng')
    report={'step':step,'rank':rank,'backend':'native_ddp','model_adam_rng_bitwise_equal':True,
            'has_fp32_master_copy':False}
    (Path(trainer.config.output_dir)/f'native_restore_rank{rank}.json').write_text(json.dumps(report,indent=2))


def restore_rng_strict(path,rank):
    """Accelerate 1.5 may suppress RNG-load errors; our own trusted state must restore."""
    rng=torch.load(Path(path)/f'random_states_{rank}.pkl',map_location='cpu',weights_only=False)
    random.setstate(rng['random_state'])
    np.random.set_state(rng['numpy_random_seed'])
    torch.set_rng_state(rng['torch_manual_seed'])
    torch.cuda.set_rng_state_all(rng['torch_cuda_manual_seed'])


def discard_auxiliary(policy):
    """Call only after strict checkpoint loading; preserve original action modules."""
    if policy.foresight.enabled or policy.point_track.version!='pri4r_embedding_v1':
        raise ValueError('Auxiliary discard only supports the original-prefix Pri4R model')
    count=sum(p.numel() for p in policy.point_track_head.parameters())
    del policy.point_track_head
    return count
