"""Reuse the original r2 training environment, dispatch the eval-only revision."""
from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
assert ROOT.name=='starVLA_pi05_embedding_position_v3_debug_20261020'
original_exec=os.execve
def dispatch(python,argv,env):
    assert argv==[python,str(ROOT/'controller.py')]
    env['PYTHONFAULTHANDLER']='1'
    original_exec(python,[python,str(HERE/'controller.py')],env)
os.execve=dispatch
sys.argv=[str(ROOT/'bootstrap.py')]
runpy.run_path(str(ROOT/'bootstrap.py'),run_name='__main__')
