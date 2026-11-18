"""Single-host JAX FSDP training with a process-parallel host DataLoader."""
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

def require_gpu_topology(jax,cfg):
    if jax.process_count()!=1 or jax.device_count()!=cfg['gpus'] or len(jax.local_devices())!=cfg['gpus']:
        raise ValueError('This single-host entry requires one process with all configured GPUs visible')
    if any(d.platform!='gpu' for d in jax.devices()):raise ValueError('Production entry requires actual GPUs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-from',type=Path,help='Complete EXP-007 Orbax step directory, read only')
    p.add_argument('--run-root',type=Path,help='Destination run directory inside the configured workspace')
    p.add_argument('--loader-workers',type=int,help='Override DataLoader processes for this launch')
    p.add_argument('--stop-after',type=int);args=p.parse_args()
    cfg=load(args.config)
    if args.loader_workers is not None:
        if args.loader_workers<0:p.error('--loader-workers must be nonnegative')
        cfg['reader']['num_workers']=args.loader_workers
    scope=Path(cfg['workspace_root']).resolve()
    configured_root=Path(cfg['run_root']).resolve()
    root=(args.run_root or configured_root).resolve()
    root.relative_to(scope)
    if root==scope:raise ValueError('Select a run directory below the workspace root')
    if args.resume_from is not None:
        args.resume_from=args.resume_from.resolve()
        args.resume_from.relative_to(scope)
        if root.is_relative_to(args.resume_from):
            raise ValueError('Destination cannot be inside the source checkpoint')
    cfg['run_root']=str(root)
    if args.stop_after is not None and cfg['mode']!='smoke':raise ValueError('stop-after is allowed only for isolated smoke runs')
    target=args.stop_after if args.stop_after is not None else cfg['num_train_steps']
    if not 1<=target<=cfg['num_train_steps']:raise ValueError('Invalid stop target')
    if cfg['mode']=='smoke' and not args.dry_run and args.stop_after is None:
        raise ValueError('Smoke requires an explicit absolute --stop-after step')
    if args.dry_run:
        print(json.dumps({'mode':cfg['mode'],'gpus':cfg['gpus'],'per_device_batch':cfg['per_device_batch_size'],'accumulation':1,'global_batch':cfg['global_batch_size'],
                          'stop':target,'schedule':cfg['learning_rate'],'execution':'one Python process; native JAX FSDP',
                          'identity':identity(cfg),'loader_workers':cfg['reader']['num_workers'],
                          'run_root':str(root),'resume_from':str(args.resume_from) if args.resume_from else None,
                          'gpu_acceptance':f"required on actual {cfg['gpus']} GPUs before formal run"}));return
    from environment import setup
    setup(cfg)
    import jax
    import numpy as np
    from openpi.training import sharding
    from data import DemoStream,Processor
    from input_pipeline import Windows,batches
    from engine import make_config,train_step,prepare_donation
    from official_init import init_train_state
    import checkpoints
    require_gpu_topology(jax,cfg)
    with writer_lock(root):
        claim=root/'CLAIM.json'
        previous=json.loads(claim.read_text()) if claim.exists() else None
        selected,marker,skipped=checkpoints.select_resume(root,cfg,args.resume_from,previous)
        if marker is not None and marker['step']>target:
            raise ValueError(f"Checkpoint step {marker['step']} exceeds stop target {target}")
        if cfg['mode']=='smoke' and target-(marker['step'] if marker else 0)>20:
            raise ValueError('A smoke invocation can run at most 20 additional updates')
        verify_assets(cfg,include_base=selected is None)
        if selected is None:
            data_receipt=json.loads((configured_root/'DATA_ACCEPTED.json').read_text())
            if not data_receipt['passed'] or data_receipt.get('real_global_batch')!=cfg['global_batch_size']:
                raise ValueError('Run check-data for the requested batch size before starting from base')
        resolved=root/'CONFIG.json'
        if not resolved.exists():
            with resolved.open('x') as f:json.dump(cfg,f,indent=2)
        if previous is None or (args.resume_from is not None and previous.get('resume_from') is None):
            record=previous if previous is not None else {'identity':identity(cfg),'created_utc':datetime.now(timezone.utc).isoformat()}
            record['resume_from']=str(args.resume_from) if args.resume_from else None
            temp_claim=claim.with_suffix('.tmp')
            temp_claim.write_text(json.dumps(record))
            temp_claim.replace(claim)
        if (root/'FINISHED.json').exists():
            done=json.loads((root/'FINISHED.json').read_text())
            if marker is None or done['step']>marker['step']:raise ValueError('Invalid completion state')
            if done['step']==cfg['num_train_steps']:
                checkpoints.verify(Path(done['export']),cfg)
                print('ALREADY_FINISHED');return
        mesh=sharding.make_mesh(cfg['fsdp_devices'])
        rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        conf=make_config(cfg);rng,init_rng=jax.random.split(jax.random.key(cfg['seed']))
        state,state_shard=init_train_state(conf,init_rng,mesh,resume=selected is not None)
        cursor=0
        if selected is not None:state,rng,cursor=checkpoints.restore(selected,state,state_shard,rng,cfg,mesh,verified_marker=marker)
        state=prepare_donation(state,state_shard)
        jax.block_until_ready(state)
        start=int(state.step)
        size = cfg['global_batch_size']
        attempt={'utc':datetime.now(timezone.utc).isoformat(),'start_step':start,'next_sample':cursor,'identity':identity(cfg),
                 'restored_from':None if selected is None else str(selected),'config_path':str(Path(args.config).resolve()),'skipped_checkpoints':skipped,
                 'devices':[str(d) for d in jax.devices()],'global_batch':size,'fsdp_devices':cfg['fsdp_devices'],
                 'loader_workers':cfg['reader']['num_workers'], 'loader_chunk':cfg['reader'].get('loader_chunk',4),
                 'prefetch_batches':cfg['reader'].get('prefetch_batches',1)}
        with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(attempt)+'\n')
        print('TRAINING_ATTEMPT',json.dumps(attempt),flush=True)
        if selected is None or (start==target and selected.parent!=(root/'checkpoints').resolve()):
            checkpoints.save(root/'checkpoints',state,rng,cursor,cfg)
        def compile_step(detailed):
            return jax.jit(functools.partial(train_step, conf, detailed_metrics=detailed),
                           in_shardings=(rep, state_shard, data_shard),
                           out_shardings=(state_shard, rep), donate_argnums=(1,))
        step_functions = {flag: compile_step(flag) for flag in (False, True)}
        stream=DemoStream(cfg);processor=Processor(cfg)
        windows = Windows(stream)
        del stream
        input_timing = {}
        batch_iter = batches(windows, processor, cfg, cursor, cursor+(target-start)*size, timing=input_timing)
        with contextlib.closing(batch_iter):
            for step in range(start + 1, target + 1):
                wall_start = time.monotonic()
                batch = next(batch_iter)
                data_ready = time.monotonic()
                if batch[1].shape[0] != size:
                    raise ValueError('Wrong global batch size')
                batch = jax.device_put(batch, data_shard)
                dispatched = time.monotonic()
                # One normal executable for long runs. Full parameter-tree diagnostics
                # belong to explicitly instrumented smoke runs, not every update.
                detailed = cfg['mode'] == 'smoke'
                with sharding.set_mesh(mesh):
                    state, metrics = step_functions[detailed](rng, state, batch)
                jax.block_until_ready((state,metrics))
                values=jax.tree.map(lambda x:float(np.asarray(x)),metrics)
                if not all(np.isfinite(x) for x in jax.tree.leaves(values)):raise FloatingPointError('Nonfinite loss/gradient/update')
                if detailed and not all(x>0 for x in values['gradient_groups'].values()):raise ValueError('A model group has no gradient')
                cursor += size
                updated = time.monotonic()
                if step==1 or step%cfg['save_interval']==0 or step==target:
                    _,receipt=checkpoints.save(root/'checkpoints',state,rng,cursor,cfg)
                    print('CHECKPOINT_COMPLETE',json.dumps(receipt|{'files':len(receipt['files'])}),flush=True)
                row={'step':step, 'next_sample':cursor, 'actual_batch_size':size,
                     'data_wait_and_pack_s':data_ready-wall_start,'h2d_dispatch_s':dispatched-data_ready,
                     'seconds':updated-dispatched,'checkpoint_s':time.monotonic()-updated,
                     'wall_s':time.monotonic()-wall_start,**input_timing,**values}
                with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if step==start+1 or step%cfg['log_interval']==0:print('UPDATE',json.dumps(row),flush=True)
        export=checkpoints.export(root/'exports',state,cfg)
        result={'identity':identity(cfg),'step':int(state.step),'next_sample':cursor,'export':str(export),'formal_complete':int(state.step)==cfg['num_train_steps']}
        name='FINISHED.json' if result['formal_complete'] else f'SMOKE_STOP_{int(state.step):06d}.json'
        (root/name).write_text(json.dumps(result,indent=2));print('RUN_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
