from pathlib import Path
import json,os,sys
from gdb_runner import run_debugged
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
env=dict(os.environ)
if env['REFERENCE_EVAL_CASE']=='baseline_reference':
    env['LD_PRELOAD']='/lib/x86_64-linux-gnu/libffi.so.7'
    env['PYTHONPATH']=':'.join([str(ROOT/'sim_code/robocasa365'),str(ROOT/'sim_code/robosuite'),env.get('PYTHONPATH','')])
target=['/root/anaconda3/envs/robocasa365/bin/python',str(ROOT/'eval_repair_r4_20261022/audited_entry.py'),'sim',*sys.argv[1:]]
proof=run_debugged(target,env,Path(env['EVAL_AUDIT_ATTEMPT'])/'native',1200)
print('NATIVE_CAPTURE '+json.dumps({k:proof.get(k) for k in ('signal','exit_code','timeout_triggered')}),flush=True)
sys.exit(0 if proof.get('exit_code')==0 and not proof.get('signal') and not proof['timeout_triggered'] else 86)
