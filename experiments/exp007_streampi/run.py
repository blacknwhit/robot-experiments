"""Single-host, eight-device JAX FSDP training. No torchrun/DDP wrapper."""
import argparse
import contextlib
import functools
import json
import os
from pathlib import Path
import time
from datetime import datetime,timezone
from contract import load,identity,verify_assets

@contextlib.contextmanager
def writer_lock(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'writer.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-after',type=int);args=p.parse_args()
    cfg=load(args.config)
    if args.stop_after is not None and cfg['mode']!='smoke':raise ValueError('stop-after is allowed only for isolated smoke runs')
    target=args.stop_after if args.stop_after is not None else cfg['num_train_steps']
    if not 1<=target<=cfg['num_train_steps']:raise ValueError('Invalid stop target')
    if cfg['mode']=='smoke' and not args.dry_run and (args.stop_after is None or target>20):
        raise ValueError('Smoke requires explicit --stop-after in 1..20; use the GPU-gated formal config for 70k')
    if args.dry_run:
        print(json.dumps({'mode':cfg['mode'],'gpus':8,'per_device_batch':8,'accumulation':1,'global_batch':64,
                          'stop':target,'schedule':cfg['learning_rate'],'execution':'one Python process; native JAX FSDP8',
                          'identity':identity(cfg),'gpu_acceptance':'required on actual 8 GPUs before formal run'}));return
    from environment import setup
    setup(cfg)
    import jax
    import numpy as np
    from openpi.training import sharding
    from data import DemoStream,Processor,fingerprint
    from engine import make_config,train_step,prepare_donation
    from official_init import init_train_state
    import checkpoints
    from concurrent.futures import ThreadPoolExecutor
    if jax.process_count()!=1 or jax.device_count()!=8 or len(jax.local_devices())!=8:
        raise ValueError('Requires one host/process with exactly 8 visible devices; do not wrap in torchrun')
    if any(d.platform!='gpu' for d in jax.devices()):raise ValueError('Production entry requires actual GPUs')
    verify_assets(cfg)
    root=Path(cfg['run_root'])
    data_receipt=json.loads((root/'DATA_ACCEPTED.json').read_text())
    if not data_receipt['passed'] or data_receipt['identity']!=identity(cfg):raise ValueError('Run check-data for this exact config first')
    with writer_lock(root):
        resolved=root/'CONFIG.json'
        if resolved.exists():
            if json.loads(resolved.read_text())!=cfg:raise ValueError('Resolved run config changed')
        else:
            with resolved.open('x') as f:json.dump(cfg,f,indent=2)
        claim=root/'CLAIM.json'
        previous=json.loads(claim.read_text()) if claim.exists() else None
        if previous is not None and previous['identity']!=identity(cfg):raise ValueError('Run directory belongs to a different recipe/source')
        selected,marker,skipped=checkpoints.discover(root/'checkpoints',cfg)
        if previous is not None and selected is None:
            raise RuntimeError('Existing run has no complete compatible checkpoint. Refusing silent restart from base; investigate and use a new isolated run if initialization never completed.')
        if previous is None:
            with claim.open('x') as f:json.dump({'identity':identity(cfg),'created_utc':datetime.now(timezone.utc).isoformat()},f)
        if (root/'FINISHED.json').exists():
            done=json.loads((root/'FINISHED.json').read_text())
            if done['identity']!=identity(cfg) or marker is None or marker['step']!=70000:raise ValueError('Invalid completion state')
            checkpoints.verify(Path(done['export']),cfg)
            print('ALREADY_FINISHED');return
        mesh=sharding.make_mesh(cfg['fsdp_devices'])
        rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        conf=make_config(cfg);rng,init_rng=jax.random.split(jax.random.key(cfg['seed']))
        state,state_shard=init_train_state(conf,init_rng,mesh,resume=selected is not None)
        cursor=0
        if selected is not None:state,rng,cursor=checkpoints.restore(selected,state,state_shard,rng,cfg,mesh)
        state=prepare_donation(state,state_shard)
        jax.block_until_ready(state)
        start=int(state.step)
        if cursor!=start*64:raise ValueError('Data cursor differs from completed global updates')
        attempt={'utc':datetime.now(timezone.utc).isoformat(),'start_step':start,'next_sample':cursor,'identity':identity(cfg),
                 'restored_from':None if selected is None else str(selected),'skipped_checkpoints':skipped,
                 'devices':[str(d) for d in jax.devices()],'global_batch':64,'fsdp_devices':8}
        with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(attempt)+'\n')
        print('TRAINING_ATTEMPT',json.dumps(attempt),flush=True)
        if selected is None:checkpoints.save(root/'checkpoints',state,rng,cursor,cfg)
        step_fn=jax.jit(functools.partial(train_step,conf),in_shardings=(rep,state_shard,data_shard),
                         out_shardings=(state_shard,rep),donate_argnums=(1,))
        stream=DemoStream(cfg);processor=Processor(cfg)
        def next_batch(index):
            # A single prefetch reader owns its caches. Only committed optimizer
            # updates advance cursor, so discarded prefetched data are replayed.
            with jax.default_device(jax.devices('cpu')[0]):
                batch=processor.batch([stream.sample(index+i) for i in range(64)])
                return jax.tree.map(np.asarray,batch)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(next_batch,cursor) if start<target else None
            while int(state.step)<target:
                batch=future.result();batch_hash=fingerprint(*batch)
                future=pool.submit(next_batch,cursor+64) if int(state.step)+1<target else None
                batch=jax.device_put(batch,data_shard);begin=time.monotonic()
                with sharding.set_mesh(mesh):state,metrics=step_fn(rng,state,batch)
                jax.block_until_ready(state)
                values=jax.tree.map(lambda x:float(np.asarray(x)),metrics)
                if not all(np.isfinite(x) for x in jax.tree.leaves(values)):raise FloatingPointError('Nonfinite loss/gradient/update')
                if not all(x>0 for x in values['gradient_groups'].values()):raise ValueError('A model group has no gradient')
                cursor+=64;step=int(state.step)
                row={'step':step,'next_sample':cursor,'input_sha256':batch_hash,'seconds':time.monotonic()-begin,**values}
                with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if step==start+1 or step%cfg['log_interval']==0:print('UPDATE',json.dumps(row),flush=True)
                if step==1 or step%cfg['save_interval']==0 or step==target:
                    _,receipt=checkpoints.save(root/'checkpoints',state,rng,cursor,cfg)
                    print('CHECKPOINT_COMPLETE',json.dumps(receipt|{'files':len(receipt['files'])}),flush=True)
        export=checkpoints.export(root/'exports',state,cfg)
        result={'identity':identity(cfg),'step':int(state.step),'next_sample':cursor,'export':str(export),'formal_complete':int(state.step)==70000}
        name='FINISHED.json' if result['formal_complete'] else f'SMOKE_STOP_{int(state.step):06d}.json'
        (root/name).write_text(json.dumps(result,indent=2));print('RUN_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
