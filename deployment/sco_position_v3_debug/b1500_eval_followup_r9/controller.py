"""One bounded dependent evaluation; never submits jobs or recovers training."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,sys,time,traceback,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
TRAIN=ROOT/'b_short_split_r8_20261023';EVAL=ROOT/'fslgpu_acceptance_r7_20261023'
RUN=ROOT/'home/engineering_runs/embedding3d_temporal_b_short_r2'
RELEASE='65c003c6766692a299eed7f58d075bbc7060e1ed8c0338f8bfec4c198a3c0dbb'
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def receipt_valid(receipt):
    return receipt.get('passed') is True and receipt.get('step')==1500 and receipt.get('release_sha256')==RELEASE and receipt.get('run')==str(RUN)
def command(out):
    return [sys.executable,str(EVAL/'evaluate.py'),'--variant','embedding3d_temporal_b',
        '--config',str(ROOT/'configs/embedding3d_temporal_b_short_1500.yaml'),
        '--checkpoint',str(RUN/'checkpoints/steps_1500_model.safetensors'),'--output',str(out),'--allocator','default']
def verify():
    for directory,expected in [(ROOT,RELEASE),(EVAL,'cac4ae92d32f9ecded7d0304db6b4c3c28233294ce9eba4f55dc2ec955f606af'),
                               (TRAIN,'2ba724ba08746c911b742b39fbd59ec0383805e65ce905a3f60a5a981e725caa')]:
        filename='release_manifest.json' if directory==ROOT else 'manifest.json'
        assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==expected
        for rel,sha in json.loads((directory/filename).read_text())['files'].items():
            assert hashlib.sha256((directory/rel).read_bytes()).hexdigest()==sha,rel
    for rel,sha in json.loads((HERE/'manifest.json').read_text())['files'].items():assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest()==sha
def main():
    import fcntl
    if '--preflight' in sys.argv:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';verify()
        out=EVAL/('b1500_cli_preflight_'+str(time.time_ns()))
        subprocess.run(command(out)+['--dry-run'],check=True)
        print('B1500_DEPENDENT_EVAL_CPU_PREFLIGHT_PASSED');return
    control=HERE/'control';control.mkdir(exist_ok=True)
    with (control/'evaluation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (control/'TERMINAL.json').exists():return
        attempt={'attempt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
                 'hostname':socket.gethostname(),'pid':os.getpid(),'started_unix':time.time(),'training_started':False}
        save(control/'current_attempt.json',attempt)
        def phase(name,**extra):save(control/'phase.json',{**attempt,'phase':name,'time_unix':time.time(),**extra})
        try:
            verify();phase('waiting_for_complete_B1500');deadline=time.monotonic()+4*3600
            while True:
                if time.monotonic()>deadline:raise TimeoutError('Four-hour bounded evaluation dependency wait expired')
                if (TRAIN/'control/HOLD_FOR_MAIN.json').exists():raise RuntimeError('B training is held; no evaluation or training retry attempted')
                marker=TRAIN/'control/B_SHORT1500_COMPLETE.json'
                if marker.exists():
                    training=json.loads(marker.read_text())
                    if not receipt_valid(training):raise ValueError('Foreign training completion receipt')
                    if not (RUN/'checkpoints/checkpoint-1500/PORTABLE_COMPLETE.json').is_file():raise ValueError('Missing completed-save marker')
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    if not apps:break
                    phase('waiting_for_idle_fslgpu_after_B1500')
                time.sleep(30)
            with (ROOT/'control/pipeline.lock').open('r') as training_lock:
                fcntl.flock(training_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
                verify()
                hardware=json.loads(subprocess.check_output([sys.executable,str(ROOT/'eval_repair_r4_20261022/gpu_identity.py'),'--egl'],text=True))
                assert len(hardware['cuda'])==1 and 'H100' in hardware['cuda'][0]['name']
                assert hardware['cuda'][0]['uuid']==hardware['egl'][0]['uuid']
                out=EVAL/('final_B1500_'+attempt['attempt']);out.mkdir();phase('B1500_EGL_evaluation',output=str(out))
                save(out/'hardware.json',hardware)
                with (out/'driver.log').open('x') as log:
                    child=subprocess.Popen(command(out),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=child.wait(timeout=1800)
                    except BaseException:
                        sys.path.insert(0,str(EVAL));from gdb_runner import stop_owned_tree
                        stop_owned_tree(child);raise
                if code:raise RuntimeError('B1500 evaluation failed; preserved '+str(out))
                gate=json.loads((out/'SMOKE_ACCEPTED.json').read_text());assert gate['application_smoke_passed']
                data=json.loads((out/'smoke/CloseBlenderLid.json').read_text())
                native=json.loads((out/'runtime_audit/native/native.json').read_text())
                sim=json.loads((out/'runtime_audit/sim_0.json').read_text())
                assert native.get('exit_code')==0 and not native.get('signal') and not native.get('timeout_triggered')
                assert sim['completed'] and data['num_trials']==2 and len(set(data['episode_trace_sha256']))==2
                assert sim['actual_egl_uuid']==hardware['cuda'][0]['uuid']
                assert data['policy_server_metadata']['variant']=='embedding3d_temporal_b'
                assert data['policy_server_metadata']['strict_full_state']['all_loaded_values_exact_at_model_dtype']
                A=json.loads((EVAL/'control/FSLGPU_EVALUATION_ACCEPTED.json').read_text());assert A['passed']
                proof={**attempt,'passed':True,'both_short_steps':1500,'full_benchmark':False,'ACP_bug_fixed':False,
                       'training_receipt':training,'A_evaluation':next(x for x in A['records'] if x['case']=='a_default'),
                       'B_evaluation':{'output':str(out),'result':data,'simulator':sim}}
                save(control/'FSLGPU_EVALUATION_ACCEPTED.json',proof);phase('AB_short_validation_complete')
                save(control/'TERMINAL.json',{**proof,'outcome':'AB_short_validation_complete'})
        except Exception:
            save(control/'TERMINAL.json',{**attempt,'outcome':'hold','traceback':traceback.format_exc(),'training_started':False});traceback.print_exc()
if __name__=='__main__':main()
