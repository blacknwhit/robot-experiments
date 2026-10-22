from pathlib import Path
import json,os,sys
from gdb_runner import run_debugged
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
out=Path(os.environ['EVAL_AUDIT_ATTEMPT'])/'native'
target=['/root/anaconda3/envs/robocasa365/bin/python',
        str(ROOT/'eval_repair_r4_20261022/audited_entry.py'),'sim',*sys.argv[1:]]
proof=run_debugged(target,dict(os.environ),out,900)
print('NATIVE_CAPTURE '+json.dumps({k:proof.get(k) for k in ('signal','exit_code','timeout_triggered','inferior_pid')}),flush=True)
sys.exit(0 if proof.get('exit_code')==0 and not proof.get('signal') and not proof['timeout_triggered'] else 86)
