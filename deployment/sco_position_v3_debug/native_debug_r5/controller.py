"""At most three read-only A1500 native-debug replays; never train or clear HOLD."""
from pathlib import Path
import datetime,fcntl,json,os,socket,subprocess,sys,time,traceback,uuid
from common_debug import HERE,ROOT,verify,command,native_summary
from gdb_runner import prerequisites
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
    control=HERE/'control';control.mkdir(exist_ok=True)
    owner=(control/'diagnostic.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    training_lock=(ROOT/'control/pipeline.lock').open('a+');fcntl.flock(training_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    if (control/'TERMINAL.json').exists():print('DIAGNOSTIC_ALREADY_TERMINAL');return
    verify()
    attempt=dict(attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
        hostname=socket.gethostname(),pid=os.getpid(),started_unix=time.time(),training_started=False,
        scope='at most three diagnostic replays of existing A1500 weights; no training or HOLD clearance')
    directory=control/'attempts'/attempt['attempt'];directory.mkdir(parents=True)
    save(control/'current_attempt.json',attempt);save(directory/'attempt.json',attempt)
    def phase(name,**extra):
        proof=dict(attempt,phase=name,time_unix=time.time(),**extra)
        save(control/'phase.json',proof);print('NATIVE_DIAGNOSTIC_PHASE '+json.dumps(proof),flush=True)
    try:
        phase('debugger_cpu_preflight');prerequisites()
        import torch
        assert torch.cuda.device_count()==2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2))
        hardware=json.loads(subprocess.check_output([sys.executable,str(ROOT/'eval_repair_r4_20261022/gpu_identity.py'),'--egl'],text=True))
        save(directory/'hardware.json',hardware)
        allowed={x['uuid'] for x in hardware['cuda']}
        egl_zero=next(x for x in hardware['egl'] if x['egl_index']==0)
        assert egl_zero.get('uuid') in allowed,'Default EGL device is outside the two allocated CUDA devices'
        phase('hardware_identity_verified',hardware=hardware)
        budget_path=control/'replay_budget.json'
        budget=json.loads(budget_path.read_text()) if budget_path.exists() else {'reserved':[],'completed':[]}
        while len(budget['reserved'])<3:
            number=len(budget['reserved'])+1;out=directory/f'replay_{number}'
            budget['reserved'].append(dict(number=number,attempt=attempt['attempt'],output=str(out),time=time.time()))
            save(budget_path,budget);out.mkdir()
            phase('native_debug_replay',replay_number=number,output=str(out))
            with (out/'driver.log').open('x') as log:
                process=subprocess.run(command(out),stdout=log,stderr=subprocess.STDOUT)
            native=native_summary(out)
            record=dict(number=number,output=str(out),returncode=process.returncode,native=native)
            budget['completed'].append(record);save(budget_path,budget)
            if native and not native.get('timeout_triggered') and native.get('signal') in ('SIGABRT','SIGSEGV','SIGBUS','SIGILL'):
                phase('native_signal_captured',**record)
                save(control/'TERMINAL.json',dict(attempt,outcome='native_signal_captured',record=record,
                    bug_fixed=False,training_started=False));return
            if process.returncode or not (out/'SMOKE_ACCEPTED.json').exists():
                raise RuntimeError('Diagnostic replay failed without a native signal: '+json.dumps(record))
        phase('bounded_replays_finished',completed=len(budget['completed']))
        outcome='three_replays_completed_no_abort' if len(budget['completed'])==3 else 'budget_exhausted_incomplete'
        save(control/'TERMINAL.json',dict(attempt,outcome=outcome,
            budget=budget,bug_fixed=False,training_started=False))
    except Exception as exc:
        phase('diagnostic_hold',error=repr(exc))
        save(control/'TERMINAL.json',dict(attempt,outcome='diagnostic_hold',error=repr(exc),traceback=traceback.format_exc(),
            bug_fixed=False,training_started=False));traceback.print_exc()
if __name__=='__main__':main()
