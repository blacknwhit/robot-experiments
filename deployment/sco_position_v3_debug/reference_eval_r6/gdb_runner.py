"""Bounded native backtrace capture, preserving the inferior's runtime env."""
from pathlib import Path
import json,os,signal,subprocess,time
HERE=Path(__file__).resolve().parent

def stop_owned_tree(proc):
    if proc.poll() is not None:return
    rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
    parents={int(a):int(b) for a,b in (line.split() for line in rows.splitlines())}
    owned={proc.pid}
    while True:
        children={pid for pid,parent in parents.items() if parent in owned}
        if children<=owned:break
        owned|=children
    for pid in sorted(owned-{proc.pid},reverse=True):
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    proc.kill();proc.wait(timeout=10)

def run_debugged(target,target_env,out,timeout=900):
    out=Path(out);out.resolve().relative_to(HERE);out.mkdir(parents=True,exist_ok=True)
    prefix=HERE.parent/'native_debug_r5_20261023/tools/root';gdb=prefix/'usr/bin/gdb'
    dbg_env=dict(target_env,LD_LIBRARY_PATH=str(prefix/'usr/lib/x86_64-linux-gnu')+':'+str(prefix/'lib/x86_64-linux-gnu'),
                 PYTHONHOME=str(prefix/'usr'),PYTHONPATH='')
    dbg_env.pop('LD_PRELOAD',None)
    result=out/'native.json'
    if result.exists():raise RuntimeError('Native capture directory already used')
    commands=['set pagination off','set confirm off','set print thread-events off',
              'set disable-randomization off','set auto-load off',
              'handle SIGPIPE nostop noprint pass','handle SIGABRT stop print nopass',
              'handle SIGSEGV stop print nopass','handle SIGBUS stop print nopass']
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME','PYTHONPATH'):
        if key in target_env:
            value=target_env[key]
            if '\n' in value:raise ValueError('Newline in runtime environment')
            commands.append('set environment '+key+' '+value)
        else:commands.append('unset environment '+key)
    commands+=['python','import gdb,json,time','capture={"started":time.time()}','def stopped(event):',
        '    if isinstance(event,gdb.SignalEvent): capture["signal"]=event.stop_signal',
        'def exited(event): capture["exit_code"]=getattr(event,"exit_code",None)',
        'gdb.events.stop.connect(stopped)','gdb.events.exited.connect(exited)','end','run','python',
        'capture["inferior_pid"]=gdb.selected_inferior().pid',
        'for command in ("info program","thread apply all bt 40","info sharedlibrary","info registers"):',
        '    try: capture[command]=gdb.execute(command,to_string=True)',
        '    except Exception as exc: capture[command]=str(exc)',
        'capture["ended"]=time.time()',
        'with open('+repr(str(result))+',"w") as stream: json.dump(capture,stream,indent=2)',
        'if gdb.selected_inferior().pid: gdb.execute("kill")','end','quit']
    script=out/'capture.gdb';script.write_text('\n'.join(commands)+'\n')
    argv=[str(gdb),'--nx','--nh','--batch','--quiet','--data-directory='+str(prefix/'usr/share/gdb'),
          '-x',str(script),'--args',*target]
    timed_out=False
    with (out/'gdb.log').open('x') as log:
        proc=subprocess.Popen(argv,env=dbg_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:stop_owned_tree(proc)
        finally:
            if proc.poll() is None:stop_owned_tree(proc)
    if not result.exists():raise RuntimeError('GDB did not capture native state; see '+str(out/'gdb.log'))
    proof=json.loads(result.read_text());proof.update(gdb_returncode=proc.returncode,timeout_triggered=timed_out,
        target=target,target_environment={k:target_env.get(k) for k in
        ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTHONHOME','CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','NVIDIA_DRIVER_CAPABILITIES','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')})
    result.write_text(json.dumps(proof,indent=2));return proof

def prerequisites():
    import sys
    parent=HERE.parent;sys.path.insert(0,str(parent/'repo/scripts'));sys.path.insert(0,str(parent/'eval_repair_r4_20261022'))
    from evaluate import simulator_environment
    env=simulator_environment('embedding3d_pos_a','0',Path('/lib/x86_64-linux-gnu/libffi.so.7'))
    target='/root/anaconda3/envs/robocasa365/bin/python'
    tag='probe_'+str(time.time_ns());root=HERE/tag
    normal=run_debugged([target,'-c','print("NORMAL_EXIT")'],env,root/'normal',60)
    abort=run_debugged([target,'-c','import os;os.abort()'],env,root/'abort',60)
    interrupted=run_debugged([target,'-c','import time;time.sleep(90)'],env,root/'timeout',5)
    assert normal.get('exit_code')==0 and not normal.get('signal')
    assert abort.get('signal')=='SIGABRT' and 'abort' in abort['thread apply all bt 40']
    assert interrupted['timeout_triggered']
    for proof in (abort,interrupted):
        pid=proof.get('inferior_pid',0)
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Debugger child still alive')
    summary=dict(passed=True,time=time.time(),directory=str(root),normal_exit=normal['exit_code'],
        abort_signal=abort['signal'],native_backtrace_captured=True,timeout_cleanup_passed=True)
    (HERE/'PREREQUISITE_ACCEPTED.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':prerequisites()
