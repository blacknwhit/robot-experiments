"""Three same-worker reference comparisons; no training, no automatic acceptance."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback,uuid
from allocator_environment import CASES,case_environment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REFERENCE=ROOT.parent/'robot_experiments_eval_affinity_gpu_20261021'
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def verify():
    spec=importlib.util.spec_from_file_location('r5_contract',ROOT/'native_debug_r5_20261023/common_debug.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.verify()
    for directory,manifest in [(HERE,'manifest.json'),(REFERENCE,'deployment_manifest.json')]:
        data=json.loads((directory/manifest).read_text());files=data.get('files',data)
        for rel,sha in files.items():
            p=directory/rel;p.resolve().relative_to(ROOT.parent)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,rel
def main():
    control=HERE/'control';control.mkdir(exist_ok=True)
    with (control/'diagnostic.lock').open('a+') as lock,(ROOT/'control/pipeline.lock').open('r') as training:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(training,fcntl.LOCK_SH|fcntl.LOCK_NB)
        if (control/'TERMINAL.json').exists():return
        attempt={'attempt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
                 'hostname':socket.gethostname(),'started_unix':time.time(),'pid':os.getpid(),'training_started':False}
        prior=control/'comparison_started.json'
        if prior.exists():
            save(control/'TERMINAL.json',{**attempt,'outcome':'diagnostic_hold','reason':'Prior worker interrupted; do not mix workers or repeat comparison without inspection','bug_fixed':False});return
        save(control/'current_attempt.json',attempt);save(prior,attempt)
        directory=control/'attempts'/attempt['attempt'];directory.mkdir(parents=True)
        records=[]
        def phase(name,**extra):
            data={**attempt,'phase':name,'time_unix':time.time(),**extra};save(control/'phase.json',data)
            print('NATIVE_DIAGNOSTIC_PHASE '+json.dumps(data),flush=True)
        try:
            verify();assert (ROOT/'control/HOLD_FOR_MAIN.json').exists()
            caps=os.environ.get('NVIDIA_DRIVER_CAPABILITIES','').split(',')
            assert 'all' in caps or 'graphics' in caps,'Container launch did not request graphics'
            phase('debugger_cpu_preflight')
            cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            with (directory/'cpu_preflight.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'cpu_preflight.py')],env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            assert json.loads((HERE/'CPU_PREFLIGHT.json').read_text())['passed']
            hardware=json.loads(subprocess.check_output([sys.executable,str(ROOT/'eval_repair_r4_20261022/gpu_identity.py'),'--egl'],text=True))
            assert len(hardware['cuda'])==2 and all('H100' in x['name'] for x in hardware['cuda'])
            assert hardware['egl'][0]['uuid']==hardware['cuda'][0]['uuid']
            hardware['capabilities']=os.environ['NVIDIA_DRIVER_CAPABILITIES']
            hardware['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True)
            save(directory/'hardware.json',hardware);phase('hardware_identity_verified',hardware=hardware)
            for number,case in enumerate(CASES,1):
                out=directory/case;out.mkdir();phase('native_debug_replay',case=case,output=str(out))
                env=case_environment(os.environ,case)
                with (out/'driver.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,str(HERE/'run_case.py'),'--case',case,'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=child.wait(timeout=1800)
                    except BaseException:
                        if child.poll() is None:
                            from gdb_runner import stop_owned_tree
                            stop_owned_tree(child)
                        raise
                native=out/'runtime_audit/native/native.json'
                captured=json.loads(native.read_text()) if native.exists() else {}
                if code and (captured.get('timeout_triggered') or captured.get('signal') not in ('SIGABRT','SIGSEGV','SIGBUS','SIGILL')):
                    raise RuntimeError('Case failed without the target native fault: '+case)
                result=out/'smoke/CloseBlenderLid.json';data=json.loads(result.read_text()) if result.exists() else None
                complete=code==0 and (out/'SMOKE_ACCEPTED.json').exists()
                if complete:
                    sim=json.loads((out/'runtime_audit/sim_0.json').read_text())
                    assert sim['completed'] and sim['actual_egl_uuid']==hardware['cuda'][0]['uuid']
                    assert data['num_trials']==2 and len(set(data['episode_trace_sha256']))==2
                records.append({'case':case,'output':str(out),'returncode':code,'complete':complete,
                                'signal':captured.get('signal'),'timeout':captured.get('timeout_triggered'),
                                'successes':data.get('successes') if data else None,
                                'traces':data.get('episode_trace_sha256') if data else None})
                save(control/'comparison.json',{'attempt':attempt,'records':records})
                phase('case_complete',case=case,output=str(out),record=records[-1])
            phase('reference_comparison_complete',records=records)
            save(control/'TERMINAL.json',{**attempt,'outcome':'reference_comparison_complete','records':records,'bug_fixed':False,'requires_main_interpretation':True})
        except Exception:
            phase('diagnostic_hold',traceback=traceback.format_exc())
            save(control/'TERMINAL.json',{**attempt,'outcome':'diagnostic_hold','records':records,'traceback':traceback.format_exc(),'bug_fixed':False});traceback.print_exc()
if __name__=='__main__':main()
