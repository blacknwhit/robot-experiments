"""Fixed task shards with optional independent workers sharing each GPU."""
import argparse,concurrent.futures,json,math,os,socket,subprocess,sys,threading,time,uuid
from pathlib import Path
from common import ROOT,VARIANTS,environment,runtime,sha256,activate
TASKS=json.loads((ROOT/'assets/eval_tasks.json').read_text())

def tasks_for_part(part):
    if not 1<=part<=13:raise ValueError('part must be 1..13')
    return TASKS[(part-1)*4:min(part*4,50)]

def gpu_assignments(gpu_ids,egl_device_ids=None):
    gpus=[g.strip() for g in gpu_ids.split(',')]
    if any(not g for g in gpus) or len(set(gpus))!=len(gpus):
        raise ValueError('Provide distinct nonempty CUDA devices; use --tasks-per-gpu for sharing')
    raw_egl=gpus if egl_device_ids is None else [g.strip() for g in egl_device_ids.split(',')]
    if len(raw_egl)!=len(gpus) or any(not g.isdecimal() for g in raw_egl):
        raise ValueError('EGL IDs must be nonnegative device-list indices, one per CUDA device; use --egl-device-ids for remapped or UUID CUDA devices')
    egl=[str(int(g)) for g in raw_egl]
    if len(set(egl))!=len(egl):raise ValueError('Each physical CUDA device needs a distinct EGL mapping')
    return gpus,egl

def simulator_environment(variant,egl_device,preload=None,cpu_threads=4):
    env=environment(variant,shim=False,overlay=False,cpu_threads=cpu_threads)
    env['PYTHONPATH']=str(runtime(variant))
    # Policy-only preloads (for example NCCL) must not leak into MuJoCo.
    env.pop('LD_PRELOAD',None)
    if preload is not None:env['LD_PRELOAD']=str(preload)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(egl_device),
               PYTHONHASHSEED='7',PYTHONFAULTHANDLER='1')
    return env

def check_port_available(port):
    with socket.socket() as probe:
        # Match asyncio's POSIX server reuse policy: TIME_WAIT after a finished
        # evaluation is not a live listener. Never enable SO_REUSEPORT.
        option=getattr(socket,'SO_EXCLUSIVEADDRUSE',socket.SO_REUSEADDR)
        probe.setsockopt(socket.SOL_SOCKET,option,1)
        probe.bind(('127.0.0.1',port))

def execution_settings(a):
    settings={'tasks_per_gpu':getattr(a,'tasks_per_gpu',1),
              'policy_cpu_threads':getattr(a,'policy_cpu_threads',4),
              'sim_cpu_threads':getattr(a,'sim_cpu_threads',4)}
    if settings['tasks_per_gpu'] not in (1,2,4):raise ValueError('tasks-per-gpu must be 1, 2 or 4')
    if any(type(settings[k]) is not int or settings[k]<1 for k in ('policy_cpu_threads','sim_cpu_threads')):
        raise ValueError('CPU thread counts must be positive integers')
    return settings

def execution_identity(a):
    settings=execution_settings(a)
    if settings=={'tasks_per_gpu':1,'policy_cpu_threads':4,'sim_cpu_threads':4}:return {}
    return {'execution':{'version':'independent_task_slots_v1',**settings}}

def task_slot_plan(tasks,gpus,egl_devices,tasks_per_gpu=1,smoke=False):
    if tasks_per_gpu not in (1,2,4):raise ValueError('tasks-per-gpu must be 1, 2 or 4')
    if not tasks or not gpus or len(gpus)!=len(egl_devices) or len(gpus)>len(tasks):raise ValueError('Invalid task/GPU mapping')
    if tasks_per_gpu==1 and ((not smoke and len(gpus)!=len(tasks)) or (smoke and len(gpus) not in (1,len(tasks)))):
        raise ValueError('Default mode needs one GPU per task; use --tasks-per-gpu 2 or 4 for fewer GPUs')
    if smoke:
        if len(tasks)<tasks_per_gpu:raise ValueError('Choose a smoke part with at least tasks-per-gpu tasks')
        # Exercise the requested sharing on one physical GPU, not separate cards.
        tasks=tasks[:tasks_per_gpu];gpus=gpus[:1];egl_devices=egl_devices[:1]
    devices=list(zip(gpus,egl_devices))*tasks_per_gpu
    count=min(len(tasks),len(devices));workers=[]
    for worker in range(count):
        gpu,egl=devices[worker]
        workers.append({'gpu':gpu,'egl':egl,'jobs':[(i,tasks[i]) for i in range(worker,len(tasks),count)]})
    return workers

def execute_plan(a,plan,identity):
    barrier=threading.Barrier(len(plan),timeout=900) if a.smoke and len(plan)>1 else None
    a.smoke_server_barrier=barrier
    def run(worker):
        try:return [execute_slot(a,task,worker['gpu'],slot,identity,worker['egl']) for slot,task in worker['jobs']]
        except BaseException:
            if barrier:barrier.abort()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan)) as pool:
        futures=[pool.submit(run,worker) for worker in plan]
        return [result for future in futures for result in future.result()]

def smoke_overlap(results):
    windows=[result.get('rollout_timing',{}) for result in results]
    if any(not isinstance(w.get('started_at'),(int,float)) or not isinstance(w.get('finished_at'),(int,float)) or
           not math.isfinite(w['started_at']) or not math.isfinite(w['finished_at']) or
           not 0<w['started_at']<w['finished_at'] for w in windows):
        raise ValueError('Shared-GPU smoke requires actual rollout timestamps')
    overlap=min(w['finished_at'] for w in windows)-max(w['started_at'] for w in windows)
    if overlap<=0:raise ValueError('Shared-GPU smoke rollouts did not overlap')
    return overlap

def prepare_shared_smoke(a,gate,identity):
    a.smoke_attempt='attempt-'+uuid.uuid4().hex
    if gate.exists():
        if json.loads(gate.read_text())['identity']!=identity:raise ValueError('Use a new output directory for changed evaluation settings')
        history=a.output/'smoke'/a.smoke_attempt;history.mkdir(parents=True)
        gate.replace(history/'previous_gate.json')

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
        if identity.get('inference_backend') in ('cuda-graph','cuda-graph-pipeline'):
            for key in ('inference_backend','inference_backend_version'):
                if metadata.get(key)!=identity[key]:raise ValueError('Inference backend identity differs: '+key)
            if trials==2 and metadata.get('cuda_graph_verify') is not True:raise ValueError('CUDA Graph smoke requires every-request action comparison')
    return d

def server_backend_args(backend,smoke):
    if backend not in ('eager','cuda-graph'):raise ValueError(backend)
    if backend=='eager':return []
    return ['--inference-backend','cuda-graph']+(['--cuda-graph-verify'] if smoke else [])

def execute_slot(a,task,gpu,slot,identity,egl_device=None):
    if egl_device is None:_,devices=gpu_assignments(gpu);egl_device=devices[0]
    r=runtime(a.variant);out=a.output/('smoke' if a.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    if a.smoke and getattr(a,'smoke_attempt',None):out=out/a.smoke_attempt;out.mkdir(parents=True,exist_ok=True)
    result=out/(task['task']+'.json');receipt=out/(task['task']+'.identity.json');trials=2 if a.smoke else 50
    if result.exists():
        if not receipt.exists() or json.loads(receipt.read_text())!=identity:raise ValueError('Existing result belongs to another checkpoint/protocol')
        return validate_result(result,task,trials,a.checkpoint,identity)
    port=a.port_base+(a.part-1)*4+slot
    check_port_available(port)
    settings=execution_settings(a)
    env=environment(a.variant,cpu_threads=settings['policy_cpu_threads']);env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED='7',OPENPI_DATA_HOME=str(a.home/'cache/openpi'))
    cmd=[sys.executable,str(ROOT/'scripts/server.py'),'--variant',a.variant,'--config',str(a.config),'--checkpoint',str(a.checkpoint),'--port',str(port)]
    cmd+=server_backend_args(getattr(a,'inference_backend','eager'),a.smoke)
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
            barrier=getattr(a,'smoke_server_barrier',None)
            if barrier:barrier.wait()
            simenv=simulator_environment(a.variant,egl_device,getattr(a,'sim_ld_preload',None),settings['sim_cpu_threads'])
            print(f"[evaluation-slot] task={task['task']} port={port} policy_cuda={gpu} simulator_egl={egl_device} policy_cpu_threads={settings['policy_cpu_threads']} sim_cpu_threads={settings['sim_cpu_threads']} OPENBLAS:1",flush=True)
            cmd=[str(a.sim_python),'-m','examples.pi05_robocasa365.eval_files.simulation_env','--port',str(port),
                '--env-name','robocasa/'+task['task'],'--env-split','pretrain','--num-trials',str(trials),'--n-envs','1',
                '--max-episode-steps',str(task['horizon']),'--n-action-steps','5','--seed','7','--checkpoint',str(a.checkpoint),
                '--profile','published_v1_0_0','--protocol-comparable','0','--horizon-multiplier','1.5','--strict-determinism','1',
                '--trace-determinism','1','--renderer-backend','egl','--result-out-path',str(result)]
            cmd+=['--ld-preload',simenv.get('LD_PRELOAD','')]
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
    p.add_argument('--sim-ld-preload',type=Path,help='Explicit simulator-only ABI library; policy preloads are never inherited')
    p.add_argument('--inference-backend',choices=('eager','cuda-graph'),default='eager',help='Opt-in exact-action CUDA Graph backend; run a matching smoke first')
    p.add_argument('--tasks-per-gpu',type=int,choices=(1,2,4),default=1,help='Maximum independent task workers per GPU; batch and episode protocol remain unchanged')
    p.add_argument('--policy-cpu-threads',type=int,default=4,help='Policy OMP/MKL threads; OpenBLAS remains 1')
    p.add_argument('--sim-cpu-threads',type=int,default=4,help='Simulator OMP/MKL threads; OpenBLAS remains 1')
    p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    for k in ['home','config','checkpoint','sim_python','output']:setattr(a,k,getattr(a,k).resolve())
    if a.sim_ld_preload is not None:a.sim_ld_preload=a.sim_ld_preload.resolve(strict=True)
    tasks=tasks_for_part(a.part);gpus,egl_devices=gpu_assignments(a.gpu_ids,a.egl_device_ids)
    settings=execution_settings(a);plan=task_slot_plan(tasks,gpus,egl_devices,settings['tasks_per_gpu'],a.smoke)
    selected=sorted((slot,task) for worker in plan for slot,task in worker['jobs'])
    if a.dry_run:
        used=list(dict.fromkeys((w['gpu'],w['egl']) for w in plan))
        print(json.dumps({'tasks':[t['task'] for _,t in selected],'GPUs':[g for g,_ in used],'EGL_devices':[e for _,e in used],'trials':2 if a.smoke else 50,
            'execution':settings,'workers':[{'gpu':w['gpu'],'egl':w['egl'],'tasks':[t['task'] for _,t in w['jobs']],
                'ports':[a.port_base+(a.part-1)*4+i for i,_ in w['jobs']]} for w in plan]},indent=2));return
    import yaml
    cfg=yaml.safe_load(a.config.read_text())
    if cfg['portable']['variant']!=a.variant:raise ValueError('Config variant mismatch')
    identity={'variant':a.variant,'checkpoint_sha256':sha256(a.checkpoint),'config_sha256':sha256(a.config),
              'source_manifest_sha256':sha256(ROOT/'provenance/source_files.sha256'),'protocol':'published_v1_0_0_strict_seed7_50x50',
              'sim_freeze_sha256':sha256(a.home/'receipts/sim.freeze.txt'),
              'evaluation_entry_sha256':sha256(__file__),
              'simulator_launch':{'python':str(a.sim_python),'preload':str(a.sim_ld_preload) if a.sim_ld_preload else None}}
    if a.inference_backend=='cuda-graph':identity.update(inference_backend='cuda-graph',inference_backend_version='native_sampler_graph_v1')
    identity.update(execution_identity(a))
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
        if a.smoke and len(plan)>1:
            prepare_shared_smoke(a,gate,identity)
        results=execute_plan(a,plan,identity)
        if a.smoke:
            proof={'identity':identity,'task':selected[0][1]['task'],'tasks':[t['task'] for _,t in selected],
                'trials':2,'application_smoke_passed':True,'concurrent_workers':len(plan)}
            if len(plan)>1:proof.update(overlap_seconds=smoke_overlap(results),evidence_subdirectory='smoke/'+a.smoke_attempt)
            gate.write_text(json.dumps(proof,indent=2))
if __name__=='__main__':main()
