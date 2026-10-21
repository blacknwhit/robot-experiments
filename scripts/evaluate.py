"""One task per GPU; 12 four-GPU shards and one two-GPU shard, with a smoke gate."""
import argparse,concurrent.futures,json,os,socket,subprocess,sys,time
from pathlib import Path
from common import ROOT,VARIANTS,environment,runtime,sha256,activate
TASKS=json.loads((ROOT/'assets/eval_tasks.json').read_text())

def tasks_for_part(part):
    if not 1<=part<=13:raise ValueError('part must be 1..13')
    return TASKS[(part-1)*4:min(part*4,50)]

def gpu_assignments(gpu_ids,egl_device_ids=None):
    gpus=[g.strip() for g in gpu_ids.split(',')]
    if any(not g for g in gpus) or len(set(gpus))!=len(gpus):
        raise ValueError('Provide one distinct nonempty CUDA device per task')
    raw_egl=gpus if egl_device_ids is None else [g.strip() for g in egl_device_ids.split(',')]
    if len(raw_egl)!=len(gpus) or any(not g.isdecimal() for g in raw_egl):
        raise ValueError('EGL IDs must be nonnegative device-list indices, one per CUDA device; use --egl-device-ids for remapped or UUID CUDA devices')
    egl=[str(int(g)) for g in raw_egl]
    if len(set(egl))!=len(egl):raise ValueError('Each task needs a distinct EGL device')
    return gpus,egl

def validate_result(path,task,trials,checkpoint,identity=None):
    d=json.loads(Path(path).read_text());success=d['successes'];traces=d['episode_trace_sha256']
    expected={'env':'robocasa/'+task['task'],'env_split':'pretrain','robocasa_version':'1.0.0','profile':'published_v1_0_0',
        'seed':7,'n_envs':1,'max_episode_steps':task['horizon'],'horizon_multiplier':1.5,'n_action_steps':5,
        'num_trials':trials,'strict_determinism':True,'trace_determinism':True,'renderer_backend':'egl',
        'checkpoint':str(Path(checkpoint).resolve()),'policy_rng_mode':'stateless_request_v1','policy_seed_version':'sha256_u32_v1','protocol_comparable':False}
    for k,v in expected.items():
        if d.get(k)!=v:raise ValueError(f'{path}: {k} mismatch {d.get(k)!r} != {v!r}')
    if len(success)!=trials or any(type(x)!=bool for x in success) or d['num_successes']!=sum(success):raise ValueError('Invalid raw successes')
    if len(traces)!=trials or len(set(traces))!=trials or any(len(t)!=64 for t in traces):raise ValueError('Missing/duplicate traces')
    if identity:
        metadata=d['policy_server_metadata']
        for k in ('variant','checkpoint_sha256','config_sha256'):
            if metadata.get(k)!=identity[k]:raise ValueError('Policy handshake identity differs: '+k)
        if metadata.get('strict_full_state',{}).get('all_loaded_values_exact_at_model_dtype') is not True:raise ValueError('Strict model load missing')
    return d

def execute_slot(a,task,gpu,slot,identity,egl_device=None):
    if egl_device is None:_,devices=gpu_assignments(gpu);egl_device=devices[0]
    r=runtime(a.variant);out=a.output/('smoke' if a.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    result=out/(task['task']+'.json');receipt=out/(task['task']+'.identity.json');trials=2 if a.smoke else 50
    if result.exists():
        if not receipt.exists() or json.loads(receipt.read_text())!=identity:raise ValueError('Existing result belongs to another checkpoint/protocol')
        return validate_result(result,task,trials,a.checkpoint,identity)
    port=a.port_base+(a.part-1)*4+slot
    with socket.socket() as probe:probe.bind(('127.0.0.1',port))
    env=environment(a.variant);env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED='7',OPENPI_DATA_HOME=str(a.home/'cache/openpi'))
    cmd=[sys.executable,str(ROOT/'scripts/server.py'),'--variant',a.variant,'--config',str(a.config),'--checkpoint',str(a.checkpoint),'--port',str(port)]
    with (out/(task['task']+'.server.log')).open('w') as log:
        server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+900
            while True:
                if server.poll() is not None:raise RuntimeError(f'Server exited; see {log.name}')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=1):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Policy server startup timeout')
                    time.sleep(2)
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
            client=WebsocketClientPolicy(port=port)
            metadata=client.get_server_metadata()
            client.close()
            for k in ('variant','checkpoint_sha256','config_sha256'):
                if metadata.get(k)!=identity[k]:raise ValueError('Connected to a foreign policy server: '+k)
            simenv=environment(a.variant,shim=False,overlay=False)
            # The simulator gets only the lightweight client/wrappers, never policy Python dependencies.
            simenv['PYTHONPATH']=str(r)
            # EGL enumerates devices independently of CUDA_VISIBLE_DEVICES.
            # Keep the server's CUDA mask, but bind the simulator by EGL index.
            simenv.pop('CUDA_VISIBLE_DEVICES',None)
            simenv.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(egl_device),PYTHONHASHSEED='7')
            print(f"[evaluation-slot] task={task['task']} policy_cuda={gpu} simulator_egl={egl_device} cpu_threads=OMP:4,MKL:4,OPENBLAS:1",flush=True)
            cmd=[str(a.sim_python),'-m','examples.pi05_robocasa365.eval_files.simulation_env','--port',str(port),
                '--env-name','robocasa/'+task['task'],'--env-split','pretrain','--num-trials',str(trials),'--n-envs','1',
                '--max-episode-steps',str(task['horizon']),'--n-action-steps','5','--seed','7','--checkpoint',str(a.checkpoint),
                '--profile','published_v1_0_0','--protocol-comparable','0','--horizon-multiplier','1.5','--strict-determinism','1',
                '--trace-determinism','1','--renderer-backend','egl','--result-out-path',str(result)]
            with (out/(task['task']+'.sim.log')).open('w') as f:subprocess.run(cmd,env=simenv,stdout=f,stderr=subprocess.STDOUT,check=True)
            data=validate_result(result,task,trials,a.checkpoint,identity);receipt.write_text(json.dumps(identity,indent=2));return data
        finally:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--home',type=Path,required=True)
    for key in ['config','checkpoint','sim-python','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--part',type=int,default=1);p.add_argument('--gpu-ids',default='0,1,2,3');p.add_argument('--port-base',type=int,default=10090)
    p.add_argument('--egl-device-ids',help='Comma-separated EGL indices paired with --gpu-ids; default assumes matching CUDA/EGL enumeration')
    p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    for k in ['home','config','checkpoint','sim_python','output']:setattr(a,k,getattr(a,k).resolve())
    tasks=tasks_for_part(a.part);gpus,egl_devices=gpu_assignments(a.gpu_ids,a.egl_device_ids)
    if (not a.smoke and len(gpus)!=len(tasks)) or (a.smoke and len(gpus) not in (1,len(tasks))) or len(set(gpus))!=len(gpus):raise ValueError('One distinct GPU per task: 4 GPUs for1..12,2 for13; smoke also accepts one GPU')
    if a.smoke:tasks=tasks[:1];gpus=gpus[:1];egl_devices=egl_devices[:1]
    if a.dry_run:print(json.dumps({'tasks':[t['task'] for t in tasks],'GPUs':gpus,'EGL_devices':egl_devices,'trials':2 if a.smoke else 50},indent=2));return
    import yaml
    cfg=yaml.safe_load(a.config.read_text())
    if cfg['portable']['variant']!=a.variant:raise ValueError('Config variant mismatch')
    identity={'variant':a.variant,'checkpoint_sha256':sha256(a.checkpoint),'config_sha256':sha256(a.config),
              'source_manifest_sha256':sha256(ROOT/'provenance/source_files.sha256'),'protocol':'published_v1_0_0_strict_seed7_50x50',
              'sim_freeze_sha256':sha256(a.home/'receipts/sim.freeze.txt')}
    gate=a.output/'SMOKE_ACCEPTED.json'
    if a.summarize:
        results=[];traces=[]
        for t in TASKS:
            receipt=a.output/'formal'/(t['task']+'.identity.json')
            if json.loads(receipt.read_text())!=identity:raise ValueError('Mixed evaluation identities')
            d=validate_result(a.output/'formal'/(t['task']+'.json'),t,50,a.checkpoint,identity);results.append(d);traces+=d['episode_trace_sha256']
        if len(set(traces))!=2500:raise ValueError('Duplicate traces across tasks')
        total=sum(d['num_successes'] for d in results)
        report={'identity':identity,'successes':total,'trials':2500,'rate':total/2500,'tasks':[{k:d[k] for k in ('env','num_successes','num_trials')} for d in results]}
        (a.output/'FINAL_ACCEPTED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
    if not a.smoke and (not gate.exists() or json.loads(gate.read_text())['identity']!=identity):raise ValueError('Run one matching --smoke first')
    a.output.mkdir(parents=True,exist_ok=True)
    activate(a.variant)
    import fcntl
    with (a.output/f'.part-{a.part}-{a.smoke}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures=[pool.submit(execute_slot,a,t,g,i,identity,e) for i,(t,g,e) in enumerate(zip(tasks,gpus,egl_devices))]
            for f in futures:f.result()
        if a.smoke:gate.write_text(json.dumps({'identity':identity,'task':tasks[0]['task'],'trials':2,'application_smoke_passed':True},indent=2))
if __name__=='__main__':main()
