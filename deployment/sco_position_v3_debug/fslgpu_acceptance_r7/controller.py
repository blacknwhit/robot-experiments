"""Direct single-H100 evaluation acceptance, independent of training recovery."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,socket,subprocess,sys,time,traceback,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
CASES=(('a_original','embedding3d_pos_a','short',1500,'original'),
       ('a_default','embedding3d_pos_a','short',1500,'default'),
       ('b_default','embedding3d_temporal_b','smoke',3,'default'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def verify():
    spec=importlib.util.spec_from_file_location('prior_contract',ROOT/'reference_eval_r6_20261023/controller.py')
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(ROOT/'reference_eval_r6_20261023'));spec.loader.exec_module(module);module.verify()
    for rel,sha in json.loads((HERE/'manifest.json').read_text())['files'].items():
        p=HERE/rel;p.resolve().relative_to(HERE);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,rel
def command(case,out):
    label,variant,stage,steps,allocator=case
    group='engineering_runs' if stage=='short' else 'smoke_runs'
    run=ROOT/'home'/group/(variant+'_'+stage+'_r2')
    return [sys.executable,str(HERE/'evaluate.py'),'--variant',variant,
        '--config',str(ROOT/'configs'/f'{variant}_{stage}_{steps}.yaml'),
        '--checkpoint',str(run/'checkpoints'/f'steps_{steps}_model.safetensors'),
        '--output',str(out),'--allocator',allocator]
def main():
    control=HERE/'control';control.mkdir(exist_ok=True)
    with (control/'evaluation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (control/'TERMINAL.json').exists():return
        attempt={'attempt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
                 'hostname':socket.gethostname(),'pid':os.getpid(),'started_unix':time.time(),'training_started':False}
        directory=control/'attempts'/attempt['attempt'];directory.mkdir(parents=True);save(control/'current_attempt.json',attempt)
        records=[]
        def phase(value,**extra):
            d={**attempt,'phase':value,'time_unix':time.time(),**extra};save(control/'phase.json',d);print(json.dumps(d),flush=True)
        try:
            verify();phase('cpu_preflight')
            with (directory/'cpu_preflight.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'cpu_preflight.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            hardware=json.loads(subprocess.check_output([sys.executable,str(ROOT/'eval_repair_r4_20261022/gpu_identity.py'),'--egl'],text=True))
            assert len(hardware['cuda'])==1 and 'H100' in hardware['cuda'][0]['name']
            assert hardware['egl'][0]['uuid']==hardware['cuda'][0]['uuid']
            save(directory/'hardware.json',hardware)
            for case in CASES:
                out=directory/case[0];out.mkdir();phase('evaluation',case=case[0],output=str(out))
                with (out/'driver.log').open('x') as log:
                    child=subprocess.Popen(command(case,out),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=child.wait(timeout=1800)
                    except BaseException:
                        from gdb_runner import stop_owned_tree
                        stop_owned_tree(child);raise
                native_path=out/'runtime_audit/native/native.json';native=json.loads(native_path.read_text()) if native_path.exists() else {}
                record={'case':case[0],'variant':case[1],'step':case[3],'returncode':code,'signal':native.get('signal'),'timeout':native.get('timeout_triggered'),'output':str(out)}
                if code:
                    records.append(record);save(control/'cases.json',records)
                    if case[0]=='a_original' and native.get('signal')=='SIGABRT' and not native.get('timeout_triggered'):continue
                    raise RuntimeError('Evaluation failed; inspect '+str(out))
                gate=json.loads((out/'SMOKE_ACCEPTED.json').read_text());assert gate['application_smoke_passed']
                sim=json.loads((out/'runtime_audit/sim_0.json').read_text());assert sim['completed'] and sim['actual_egl_uuid']==hardware['cuda'][0]['uuid']
                data=json.loads((out/'smoke/CloseBlenderLid.json').read_text())
                assert data['num_trials']==2 and len(set(data['episode_trace_sha256']))==2
                assert native.get('exit_code')==0 and not native.get('signal') and not native.get('timeout_triggered')
                record.update(passed=True,successes=data['successes'],traces=data['episode_trace_sha256'],checkpoint_sha256=data['policy_server_metadata']['checkpoint_sha256'],strict_load=data['policy_server_metadata']['strict_full_state'],simulator=sim)
                records.append(record);save(control/'cases.json',records)
            bycase={r['case']:r for r in records};parity=None
            if bycase['a_original'].get('passed'):
                parity=bycase['a_original']['traces']==bycase['a_default']['traces'] and bycase['a_original']['successes']==bycase['a_default']['successes']
                assert parity,'Allocator switch changed observed trajectories'
            accepted={**attempt,'passed':True,'records':records,'A_allocator_trace_parity':parity,'ACP_bug_fixed':False,
                      'scope':'Single-H100 fslgpu A1500 and Bsmoke3 complete EGL evaluation; B1500 still pending'}
            save(control/'FSLGPU_EVALUATION_ACCEPTED.json',accepted);phase('fslgpu_evaluation_accepted')
            save(control/'TERMINAL.json',{**accepted,'outcome':'fslgpu_evaluation_accepted'})
        except Exception:
            save(control/'TERMINAL.json',{**attempt,'outcome':'hold','records':records,'traceback':traceback.format_exc(),'ACP_bug_fixed':False});traceback.print_exc()
if __name__=='__main__':main()
