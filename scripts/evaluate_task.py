"""One GPU / one task; captured scene plans allow exact n_envs comparisons."""
import argparse,json,os,socket,subprocess,sys,time,uuid
from pathlib import Path
from common import ROOT,VARIANTS,activate,environment,sha256,runtime
from evaluate import TASKS,gpu_assignments,validate_result
from episode_plan import atomic_json,load_plan,file_sha,load_plan_set,build_plan_set
from legacy_contract import require_same_policy,require_same_trajectories,historical_inventory,check_historical

def check_port_available(port):
    with socket.socket() as check:
        # Match asyncio's server behavior: TIME_WAIT is reusable, a live listener is not.
        check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check.bind(('127.0.0.1',port))

def bind_output_identity(output,identity):
    """Parallel shards share one immutable model/config/scene/run identity."""
    import fcntl
    with (output/'EVALUATION_IDENTITY.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=output/'EVALUATION_IDENTITY.json'
        if path.exists():
            if json.loads(path.read_text())!=identity:raise ValueError('Output identity differs; use a new output directory')
            return
        previous=output/'SMOKE_ACCEPTED.json'
        if previous.exists() and json.loads(previous.read_text()).get('identity')!=identity:
            raise ValueError('Output identity differs from previous smoke; use a new output directory')
        for receipt in (output/'formal').glob('*.identity.json'):
            if json.loads(receipt.read_text()).get('identity')!=identity:
                raise ValueError('Output identity differs from existing results; use a new output directory')
        atomic_json(path,identity)

def validate_rgb(data,trials,horizon):
    proof=data.get('simulation_optimization',{})
    if (proof.get('version')!='native_last_frame_rgb_v1'
        or proof.get('control_steps')!=trials*horizon
        or proof.get('rgb_reused',0)<=0 or proof.get('rgb_rendered',0)<=0):
        raise ValueError('Missing selective RGB execution evidence')

def validate_fixed(data,plan,plan_sha,trials,n_envs,identity,verify=False):
    expected={'complete':True,'env':plan['env'],'episode_protocol':'frozen_scene_plan_v1','episode_plan_sha256':plan_sha,
        'num_trials':trials,'n_envs':n_envs,'logical_environment_index':0,'episode_ids':list(range(trials)),
        'seed':7,'n_action_steps':5,'max_episode_steps':plan['max_episode_steps']}
    for k,v in expected.items():
        if data.get(k)!=v:raise ValueError('Result contract mismatch: '+k)
    successes=data['successes'];traces=data['episode_trace_sha256'];metadata=data['policy_server_metadata']
    if len(successes)!=trials or any(type(x) is not bool for x in successes) or sum(successes)!=data['num_successes']:raise ValueError('Invalid raw successes')
    if len(traces)!=trials or len(set(traces))!=trials or any(len(t)!=64 for t in traces):raise ValueError('Invalid trace coverage')
    for k in ('variant','checkpoint_sha256','config_sha256','inference_backend','inference_backend_version'):
        if metadata.get(k)!=identity[k]:raise ValueError('Policy identity mismatch: '+k)
    if metadata.get('strict_full_state',{}).get('all_loaded_values_exact_at_model_dtype') is not True:raise ValueError('Strict model load missing')
    if verify and metadata.get('cuda_graph_verify') is not True:raise ValueError('Smoke requires full action comparisons')
    scenes=data['initial_scenes']
    if [s['episode'] for s in scenes]!=list(range(trials)):raise ValueError('Missing initial scene evidence')
    for scene,row in zip(scenes,plan['resets']):
        if scene['observation_sha256']!=row['initial_observation_sha256'] or scene['state_sha256']!=row['initial_state_sha256']:raise ValueError('Scene mismatch')
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--home',type=Path,required=True)
    for k in ('config','checkpoint','sim-python','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--task',choices=[x['task'] for x in TASKS]);p.add_argument('--gpu-id',default='0');p.add_argument('--egl-device-id')
    p.add_argument('--port',type=int,default=19090);p.add_argument('--n-envs',type=int,choices=range(1,51),default=1)
    p.add_argument('--scene-protocol',choices=('legacy','shared-scenes'),default='legacy')
    p.add_argument('--historical-results',type=Path,help='Optional old raw task results; every selected trajectory must match')
    p.add_argument('--rgb-last-frame',action='store_true',help='Experimental: skip unread intermediate RGB while preserving native clocks')
    p.add_argument('--capture-plan',action='store_true',help='Run the original serial episode stream and record reusable initializations')
    p.add_argument('--plan-dir',type=Path,help='Legacy: this exact checkpoint/config own captured serial plans')
    p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    for k in ('home','config','checkpoint','sim_python','output'):setattr(a,k,getattr(a,k).resolve())
    if a.plan_dir:a.plan_dir=a.plan_dir.resolve()
    if a.historical_results:a.historical_results=a.historical_results.resolve()
    if a.scene_protocol=='shared-scenes' and a.historical_results:p.error('Shared scenes do not certify the old model-dependent scene sequence')
    if a.scene_protocol=='shared-scenes' and a.capture_plan:p.error('Shared scenes require an existing plan-dir; capture-plan is legacy only')
    if not a.plan_dir and a.scene_protocol=='legacy':a.capture_plan=True
    if a.capture_plan:
        if a.n_envs!=1:p.error('Legacy first evaluation requires --n-envs 1; multiple environments need this checkpoint own existing serial plan')
        if a.plan_dir:p.error('--capture-plan does not consume a plan-dir')
    elif not a.plan_dir:p.error('Replay requires --plan-dir')
    if a.rgb_last_frame and not a.capture_plan:p.error('--rgb-last-frame currently supports native serial capture only')
    if not a.summarize and not a.task:p.error('--task is required')
    gpu,egl=gpu_assignments(a.gpu_id,a.egl_device_id)
    if len(gpu)!=1:p.error('This entry uses exactly one GPU for one task')
    task=next((t for t in TASKS if t['task']==a.task),None)
    trials=(2 if a.capture_plan else max(2,a.n_envs)) if a.smoke else 50
    if a.dry_run:
        print(json.dumps({'task':a.task,'gpu':gpu[0],'egl':egl[0],'n_envs':a.n_envs,'trials':trials,'mode':'capture' if a.capture_plan else 'replay'},indent=2));return
    import yaml
    cfg=yaml.safe_load(a.config.read_text())
    if cfg['portable']['variant']!=a.variant:raise ValueError('Config variant mismatch')
    identity={'variant':a.variant,'checkpoint_sha256':sha256(a.checkpoint),'config_sha256':sha256(a.config),
        'source_manifest_sha256':sha256(ROOT/'provenance/source_files.sha256'),'sim_freeze_sha256':sha256(a.home/'receipts/sim.freeze.txt'),
        'inference_backend':'cuda-graph-pipeline','inference_backend_version':'cpu_gpu_pipeline_batch1_v1',
        'episode_protocol':'capture_legacy_serial_v1' if a.capture_plan else 'frozen_scene_plan_v1','n_envs':a.n_envs,
        'scene_protocol':a.scene_protocol,'rgb_last_frame':a.rgb_last_frame}
    history=None
    if a.historical_results:
        history,identity['historical_results_sha256']=historical_inventory(a.historical_results,TASKS)
        if a.task and a.task not in history:raise ValueError('Task missing from historical results')
    plan_set=None
    if not a.capture_plan:
        plan_set,identity['episode_plan_set_sha256']=load_plan_set(a.plan_dir)
        if a.task and a.task not in plan_set:raise ValueError('Task is absent from the frozen plan set')
        if a.scene_protocol=='legacy':
            for name in plan_set:require_same_policy(load_plan(a.plan_dir/(name+'.json')),identity)
    a.output.mkdir(parents=True,exist_ok=True);gate=a.output/'SMOKE_ACCEPTED.json'
    if a.summarize:
        results=[];plan_hashes={}
        for t in TASKS:
            receipt=json.loads((a.output/'formal'/(t['task']+'.identity.json')).read_text())
            if receipt['identity']!=identity:raise ValueError('Mixed evaluation identities')
            path=(a.output/'plans' if a.capture_plan else a.plan_dir)/(t['task']+'.json');plan=load_plan(path,'robocasa/'+t['task'],50)
            result_path=a.output/'formal'/(t['task']+'.json');d=json.loads(result_path.read_text())
            if a.capture_plan:validate_result(result_path,t,50,a.checkpoint,identity)
            else:
                validate_fixed(d,plan,file_sha(path),50,a.n_envs,identity)
                if a.scene_protocol=='legacy':require_same_trajectories(d,plan['reference_result'],50)
            if history:check_historical(a.historical_results,history,t['task'],d,identity,50)
            if a.rgb_last_frame:validate_rgb(d,50,t['horizon'])
            if receipt['episode_plan_sha256']!=file_sha(path):raise ValueError('Plan changed after evaluation')
            plan_hashes[t['task']]=file_sha(path);results.append(d)
        traces=[h for d in results for h in d['episode_trace_sha256']]
        if len(traces)!=2500 or len(set(traces))!=2500:raise ValueError('Coverage mismatch')
        if a.capture_plan:plan_set_sha=build_plan_set(a.output/'plans')
        else:
            if load_plan_set(a.plan_dir)!=(plan_set,identity['episode_plan_set_sha256']):raise ValueError('Plan set changed')
            plan_set_sha=identity['episode_plan_set_sha256']
        atomic_json(a.output/'FINAL_ACCEPTED.json',{'identity':identity,'episode_plan_set_sha256':plan_set_sha,'plans':plan_hashes,
            'trials':2500,'successes':sum(d['num_successes'] for d in results),'tasks':[{'env':d['env'],'successes':d['num_successes']} for d in results]});return
    if not a.smoke:
        if a.scene_protocol=='legacy' and (not gate.exists() or json.loads(gate.read_text()).get('identity')!=identity or json.loads(gate.read_text()).get('application_smoke_passed') is not True):
            raise ValueError('Run one matching task --smoke first')
    bind_output_identity(a.output,identity)
    import fcntl
    with (a.output/(a.task+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result_path=a.output/'formal'/(a.task+'.json');receipt_path=result_path.with_suffix('.identity.json')
        plan_path=(a.output/('smoke-plans' if a.smoke else 'plans') if a.capture_plan else a.plan_dir)/(a.task+'.json')
        if not a.smoke and result_path.exists():
            old=json.loads(receipt_path.read_text())
            if old['identity']!=identity or old['episode_plan_sha256']!=file_sha(plan_path):raise ValueError('Existing result identity differs')
            if a.capture_plan:validate_result(result_path,task,50,a.checkpoint,identity)
            else:
                existing=json.loads(result_path.read_text());existing_plan=load_plan(plan_path)
                validate_fixed(existing,existing_plan,file_sha(plan_path),50,a.n_envs,identity)
                if a.scene_protocol=='legacy':require_same_trajectories(existing,existing_plan['reference_result'],50)
            if history:check_historical(a.historical_results,history,a.task,json.loads(result_path.read_text()),identity,50)
            if a.rgb_last_frame:validate_rgb(json.loads(result_path.read_text()),50,task['horizon'])
            print('Already complete:',a.task);return
        attempt=a.output/('smoke' if a.smoke else 'runs')/a.task/uuid.uuid4().hex;attempt.mkdir(parents=True)
        if a.smoke and gate.exists():
            if json.loads(gate.read_text())['identity']!=identity:raise ValueError('Use a new output directory')
            gate.replace(attempt/'previous_gate.json')
        if a.capture_plan and a.smoke:plan_path=attempt/'captured-plan.json'
        if a.capture_plan and plan_path.exists():raise FileExistsError('Plan exists without a reusable result; preserve it and select a new output')
        plan=None
        if not a.capture_plan:
            plan=load_plan(plan_path,'robocasa/'+a.task,trials)
            if file_sha(plan_path)!=plan_set[a.task]:raise ValueError('Plan changed after collection validation')
            if plan['sim_freeze_sha256']!=identity['sim_freeze_sha256'] or plan['max_episode_steps']!=task['horizon']:raise ValueError('Simulator or horizon differs from scene plan')
        activate(a.variant)
        check_port_available(a.port)
        env=environment(a.variant);env.update(CUDA_VISIBLE_DEVICES=gpu[0],PYTHONHASHSEED='7',OPENPI_DATA_HOME=str(a.home/'cache/openpi'))
        cmd=[sys.executable,str(ROOT/'scripts/server.py'),'--variant',a.variant,'--config',str(a.config),'--checkpoint',str(a.checkpoint),'--port',str(a.port),'--inference-backend','cuda-graph-pipeline']
        if a.smoke:cmd.append('--cuda-graph-verify')
        with (attempt/'server.log').open('w') as log:
            server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+1200
                while True:
                    if server.poll() is not None:raise RuntimeError('Policy startup failed; see server.log')
                    try:
                        with socket.create_connection(('127.0.0.1',a.port),timeout=1):break
                    except OSError:
                        if time.monotonic()>deadline:raise TimeoutError('Policy startup timeout')
                        time.sleep(2)
                from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
                client=WebsocketClientPolicy(port=a.port);metadata=client.get_server_metadata();client.close()
                for key in ('variant','checkpoint_sha256','config_sha256','inference_backend','inference_backend_version'):
                    if metadata.get(key)!=identity[key]:raise ValueError('Foreign policy server: '+key)
                if a.smoke and metadata.get('cuda_graph_verify') is not True:raise ValueError('Smoke requires every-request exact action verification')
                simenv=environment(a.variant,shim=False,overlay=False);simenv.pop('CUDA_VISIBLE_DEVICES',None)
                simenv.update(PYTHONPATH=os.pathsep.join((str(ROOT/'scripts'),str(runtime(a.variant)))),PYTHONHASHSEED='7',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=egl[0])
                def run(label,workers,capture=False,selected_plan=None,rgb=None):
                    target=attempt/label
                    cmd=[str(a.sim_python),str(ROOT/'scripts/fixed_episode_sim.py'),'--plan',str(selected_plan or plan_path),'--output',str(target),
                        '--task','robocasa/'+a.task,'--port',str(a.port),'--trials',str(trials),'--n-envs',str(workers)]
                    if capture:cmd+=['--capture','--horizon',str(task['horizon']),'--checkpoint',str(a.checkpoint),'--sim-freeze-sha256',identity['sim_freeze_sha256']]
                    elif a.smoke:cmd.append('--barrier')
                    if (a.rgb_last_frame if rgb is None else rgb):cmd.append('--rgb-last-frame')
                    with (attempt/(label+'.log')).open('w') as log:subprocess.run(cmd,env=simenv,stdout=log,stderr=subprocess.STDOUT,check=True)
                    return json.loads((target/'result.json').read_text())
                if a.capture_plan:
                    original_plan_path=attempt/'unoptimized-plan.json'
                    if a.rgb_last_frame and a.smoke:
                        original=run('serial-original',1,True,original_plan_path,False)
                        validate_result(attempt/'serial-original/result.json',task,trials,a.checkpoint,identity)
                    result=run('capture',1,True);validate_result(attempt/'capture/result.json',task,trials,a.checkpoint,identity)
                    captured=load_plan(plan_path,'robocasa/'+a.task,trials)
                    if a.rgb_last_frame:
                        validate_rgb(result,trials,task['horizon'])
                        if a.smoke:
                            require_same_trajectories(result,original,trials)
                            if captured['resets']!=load_plan(original_plan_path)['resets']:raise ValueError('Selective RGB changed initial scenes or reset RNG')
                else:
                    reference=None
                    if a.smoke:
                        source=plan.get('reference_result',{}).get('policy_server_metadata',{})
                        if all(source.get(k)==identity[k] for k in ('variant','checkpoint_sha256','config_sha256')):
                            reference={k:plan['reference_result'][k][:trials] for k in ('successes','episode_trace_sha256')}
                        else:
                            reference=run('serial-reference',1);validate_fixed(reference,plan,file_sha(plan_path),trials,1,identity,True)
                    result=run('parallel-replay',a.n_envs);validate_fixed(result,plan,file_sha(plan_path),trials,a.n_envs,identity,a.smoke)
                    if a.scene_protocol=='legacy':require_same_trajectories(result,plan['reference_result'],trials)
                    if a.smoke:
                        if any(result[k]!=reference[k] for k in ('successes','episode_trace_sha256')):raise ValueError('Parallel trajectories differ from single-environment reference')
                        workers=[json.loads((attempt/'parallel-replay'/f'worker{i}.json').read_text()) for i in range(a.n_envs)]
                        if len(workers)!=a.n_envs or min(w['ended'] for w in workers)<=max(w['started'] for w in workers):raise ValueError('Workers did not overlap')
                if not a.capture_plan and load_plan_set(a.plan_dir)!=(plan_set,identity['episode_plan_set_sha256']):raise ValueError('Plan set changed during evaluation')
                historical_proof=check_historical(a.historical_results,history,a.task,result,identity,trials) if history else None
                if a.smoke:
                    atomic_json(gate,{'identity':identity,'task':a.task,'trials':trials,'episode_plan_sha256':file_sha(plan_path),
                        'evidence':str(attempt.relative_to(a.output)),'application_smoke_passed':True,'trajectory_parity':not a.capture_plan,
                        'historical_proof':historical_proof,'rgb_reference_checked':a.rgb_last_frame and a.capture_plan})
                else:
                    atomic_json(result_path,result);atomic_json(receipt_path,{'identity':identity,'episode_plan_sha256':file_sha(plan_path),'evidence':str(attempt.relative_to(a.output)),'historical_proof':historical_proof})
            finally:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()
if __name__=='__main__':main()
