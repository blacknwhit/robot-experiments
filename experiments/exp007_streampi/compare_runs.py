"""Compare actual full-model step3 continuation after an eight-GPU restart."""
import argparse
import dataclasses
import hashlib
import json
import gc
from pathlib import Path
from contract import load,recipe_identity

def signature(state,rng,cursor):
    import jax
    import numpy as np
    result={}
    for name,tree in [('params',state.params),('optimizer',state.opt_state),('ema',state.ema_params)]:
        h=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            a=np.asarray(value)
            h.update(str(path).encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
        result[name]=h.hexdigest()
    return result|{'step':int(state.step),'cursor':cursor,'rng':jax.random.key_data(rng).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--resumed',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    roots=[Path(a.reference),Path(a.resumed)];cfgs=[load(r/'CONFIG.json') for r in roots]
    comparable=[{k:v for k,v in c.items() if k!='run_root'} for c in cfgs]
    if comparable[0]!=comparable[1]:raise ValueError('Runs differ beyond isolated run path')
    from environment import setup
    setup(cfgs[0])
    import jax
    from openpi.training import sharding
    from official_init import init_train_state
    from engine import make_config
    import checkpoints
    if jax.device_count()!=8 or jax.process_count()!=1 or any(d.platform!='gpu' for d in jax.devices()):
        raise ValueError('This gate requires actual eight-GPU single-host topology')
    mesh=sharding.make_mesh(8);results=[]
    for root,cfg in zip(roots,cfgs):
        rng,ir=jax.random.split(jax.random.key(cfg['seed']))
        shape,ss=init_train_state(make_config(cfg),ir,mesh,resume=True)
        state,rr,cursor=checkpoints.restore(root/'checkpoints/step_000003',shape,ss,rng,cfg,mesh)
        results.append(signature(state,rr,cursor));del state,shape,ss;gc.collect()
    attempts=[json.loads(x) for x in (roots[1]/'attempts.jsonl').read_text().splitlines()]
    if not any(r['start_step']==2 and r['restored_from'] for r in attempts):raise ValueError('No actual step2 restart evidence')
    if results[0]!=results[1]:raise AssertionError({'reference':results[0],'resumed':results[1]})
    r={'passed':True,'exact_full_state':True,'reference':str(roots[0]),'resumed':str(roots[1]),'state':results[0],
       'physical_gpu_count':8,'device_kinds':[d.device_kind for d in jax.devices()],'recipe_identity':recipe_identity(cfgs[0])}
    out=Path(a.out).resolve();out.relative_to(Path(cfgs[0]['workspace_root']).resolve())
    with out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r))
if __name__=='__main__':main()
