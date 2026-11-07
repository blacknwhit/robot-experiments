"""Eight virtual CPU devices: real optimizer, FSDP, checkpoint and restart checks."""
import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['reference','resume'],required=True);args=p.parse_args()
root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_FLAGS']='--xla_force_host_platform_device_count=8'
from environment import setup
cfg={'workspace_root':str(root.parent),'run_root':str(root),'tokenizer':str(root/'unused_tokenizer'),'gpus':8,'global_batch_size':64,'deterministic_gpu':False}
setup(cfg,cpu=True)
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from types import SimpleNamespace
from openpi.training import sharding,utils
from engine import train_step,prepare_donation
from checkpoints import save,restore,discover
from compare_runs import signature
from contract import validate,ROOT

recipe=json.loads((ROOT/'recipe_8gpu64_70k.json').read_text());recipe['mode']='formal';validate(recipe)
rejected=0
for key,value in [('gpus',4),('global_batch_size',32),('gradient_accumulation_steps',8),('num_train_steps',30000),('enable_jitter',True)]:
    try:validate(recipe|{key:value})
    except AssertionError:rejected+=1
assert rejected==5
class Tiny(nnx.Module):
    def __init__(self):
        rng=nnx.Rngs(42)
        self.PaliGemma=nnx.Dict(img=nnx.Linear(2048,512,rngs=rng),llm=nnx.Linear(512,32,rngs=rng))
        self.action=nnx.Linear(32,1,rngs=rng)
    def compute_loss(self,rng,observation,actions,train=False):
        x=nnx.swish(self.PaliGemma.img(observation));x=nnx.swish(self.PaliGemma.llm(x))
        pred=self.action(x)+.01*jax.random.normal(rng,actions.shape)
        return jnp.square(pred-actions)
mesh=sharding.make_mesh(8);rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
model=Tiny();params=nnx.state(model);tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(.001,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
state=utils.TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(params),ema_decay=.999,ema_params=params)
shape=jax.eval_shape(lambda:state);ss=sharding.fsdp_sharding(shape,mesh)
conf=SimpleNamespace(trainable_filter=nnx.Param,lr_schedule=SimpleNamespace(create=lambda:lambda step:jnp.float32(.001)))
step_fn=jax.jit(functools.partial(train_step,conf),in_shardings=(rep,ss,data_shard),out_shardings=(ss,rep),donate_argnums=(1,))
rng=jax.random.key(42);cursor=0
if args.mode=='resume':
    picked,_,skipped=discover(root/'checkpoints',cfg)
    assert picked.name=='step_000002' and skipped
    state,rng,cursor=restore(picked,shape,ss,rng,cfg,mesh)
    assert int(state.step)==2 and cursor==128
else:
    state=jax.device_put(state,ss)
state=prepare_donation(state,ss)
for a,b in zip(jax.tree.leaves(state.params),jax.tree.leaves(state.ema_params)):
    for x,y in zip(a.addressable_shards,b.addressable_shards):
        assert x.data.unsafe_buffer_pointer()!=y.data.unsafe_buffer_pointer()
assert len(state.params['PaliGemma']['img']['kernel'].value.addressable_shards)==8
assert state.params['PaliGemma']['img']['kernel'].value.addressable_shards[0].data.size==2048*512//8
while int(state.step)<3:
    i=int(state.step);gen=np.random.default_rng(i)
    batch=(gen.normal(size=(64,2048)).astype(np.float32),gen.normal(size=(64,1)).astype(np.float32))
    with sharding.set_mesh(mesh):state,metrics=step_fn(rng,state,jax.device_put(batch,data_shard))
    jax.block_until_ready(state);cursor+=64
    assert np.isfinite(float(metrics['loss']))
    if int(state.step)==2 and args.mode=='reference':
        save(root/'checkpoints',state,rng,cursor,cfg)
        (root/'checkpoints/step_999999').mkdir() # Simulate an interrupted latest save.
result=signature(state,rng,cursor)
if args.mode=='reference':(root/'REFERENCE.json').write_text(json.dumps(result,indent=2))
else:
    assert result==json.loads((root/'REFERENCE.json').read_text())
    (root/'CPU_8DEVICE_ACCEPTED.json').write_text(json.dumps({'passed':True,'virtual_devices':8,'global_batch':64,'physical_gpus':0,
        'full_state_resume_exact':True,'partial_latest_skipped':True,'contract_rejections':5,'scope':'tiny model; real FSDP/AdamW/EMA/checkpoint machinery, not full-model GPU acceptance','signature':result},indent=2))
print(args.mode.upper()+'_ACCEPTED',json.dumps(result),flush=True)
