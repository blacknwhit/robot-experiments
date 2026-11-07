"""Atomic, integrity-checked full-state checkpoints with native sharding restore."""
import copy
import dataclasses
import json
from pathlib import Path
import time
import uuid
import jax
import numpy as np
import orbax.checkpoint as ocp
from contract import sha,identity

def payload(state,rng,cursor):
    return {'params':state.params.to_pure_dict(),'opt_state':state.opt_state,
            'ema_params':state.ema_params.to_pure_dict(),'step':state.step,
            'train_rng':jax.random.key_data(rng),'next_sample':np.int32(cursor)}

def file_manifest(root):
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}

def verify(path,cfg):
    marker=json.loads((path/'COMPLETE.json').read_text())
    if marker['identity']!=identity(cfg):raise ValueError('Checkpoint identity mismatch')
    if not marker['files']:raise ValueError('Empty checkpoint manifest')
    for rel,record in marker['files'].items():
        p=(path/rel).resolve();p.relative_to(path.resolve())
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise ValueError(f'Incomplete/corrupt checkpoint: {rel}')
    return marker

def discover(root,cfg):
    skipped=[]
    for path in sorted(root.glob('step_[0-9]*'),reverse=True):
        if not path.is_dir():continue
        try:return path,verify(path,cfg),skipped
        except (OSError,ValueError,KeyError) as exc:skipped.append({'path':path.name,'reason':str(exc)})
    return None,None,skipped

def save(root,state,rng,cursor,cfg):
    root.mkdir(parents=True,exist_ok=True)
    step=int(state.step);final=root/f'step_{step:06d}'
    if final.exists():raise FileExistsError(final)
    temp=root/f'.step_{step:06d}_{uuid.uuid4().hex}.partial';temp.mkdir()
    started=time.monotonic()
    with ocp.PyTreeCheckpointer() as writer:writer.save(temp/'state',payload(state,rng,cursor))
    marker={'identity':identity(cfg),'step':step,'next_sample':int(cursor),'files':file_manifest(temp),
            'save_seconds':time.monotonic()-started,'gpus':cfg['gpus'],'global_batch_size':cfg['global_batch_size']}
    (temp/'COMPLETE.json').write_text(json.dumps(marker,indent=2))
    temp.rename(final)
    return final,marker

def restore(path,shape_state,state_sharding,rng,cfg,mesh):
    marker=verify(path,cfg)
    target=payload(shape_state,rng,marker['next_sample'])
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    shard={'params':state_sharding.params.to_pure_dict(),'opt_state':state_sharding.opt_state,
           'ema_params':state_sharding.ema_params.to_pure_dict(),'step':state_sharding.step,
           'train_rng':rep,'next_sample':rep}
    args=jax.tree.map(lambda s:ocp.ArrayRestoreArgs(restore_type=jax.Array,sharding=s),shard)
    with ocp.PyTreeCheckpointer() as reader:
        loaded=reader.restore(path/'state',args=ocp.args.PyTreeRestore(item=target,restore_args=args))
    params=copy.deepcopy(shape_state.params);ema=copy.deepcopy(shape_state.ema_params)
    params.replace_by_pure_dict(loaded['params']);ema.replace_by_pure_dict(loaded['ema_params'])
    state=dataclasses.replace(shape_state,params=params,opt_state=loaded['opt_state'],ema_params=ema,step=loaded['step'])
    if int(state.step)!=marker['step'] or int(loaded['next_sample'])!=marker['next_sample']:raise ValueError('Counter mismatch')
    return state,jax.random.wrap_key_data(loaded['train_rng']),int(loaded['next_sample'])

def export(root,state,cfg):
    from openpi.shared import normalize
    root.mkdir(parents=True,exist_ok=True);step=int(state.step);final=root/f'step_{step:06d}'
    if final.exists():
        verify(final,cfg);return final
    temp=root/f'.export_{step:06d}_{uuid.uuid4().hex}.partial';temp.mkdir()
    with ocp.PyTreeCheckpointer() as writer:writer.save(temp/'params',{'params':state.ema_params.to_pure_dict()})
    normalize.save(temp/'assets',normalize.load(cfg['norm_assets']))
    (temp/'config.json').write_text(json.dumps(cfg,indent=2))
    (temp/'COMPLETE.json').write_text(json.dumps({'identity':identity(cfg),'step':step,'weights':'ema','files':file_manifest(temp)},indent=2))
    temp.rename(final);return final
