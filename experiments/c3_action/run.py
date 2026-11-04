"""One launcher, one output lock; rerun this exact command for full-state resume."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from contract import ROOT,env,validate

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--stop',type=int)
 p.add_argument('--dry-run',action='store_true');a=p.parse_args();c=json.loads(a.config.read_text());validate(c)
 stop=a.stop or c['steps']
 if c['mode']=='formal':assert stop==70000,'Formal run always targets 70k'
 else:assert 1<=stop<=10,'Use isolated smoke outputs for bounded tests'
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node',str(c['world']),
      str(Path(__file__).with_name('worker.py')),'--config',str(a.config.resolve()),'--stop',str(stop)]
 print(json.dumps({'command':cmd,'world':c['world'],'micro_batch':8,'accumulation':c['accumulation'],
    'global_batch':64,'stop':stop,'cosine_decay_steps':c['schedule_steps'],'native_full_state_resume':True},indent=2),flush=True)
 if a.dry_run:return
 import fcntl
 run=Path(c['run']);run.mkdir(parents=True,exist_ok=True)
 with (run/'writer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  subprocess.run(cmd,cwd=ROOT,env=env(c),check=True,pass_fds=(lock.fileno(),))
if __name__=='__main__':main()
