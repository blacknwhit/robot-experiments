from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
cpu='--cpu' in sys.argv
original=os.execve
def dispatch(python,argv,env):
    assert argv==[python,str(ROOT/'controller.py')]
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    original(python,[python,str(HERE/'controller.py'),*(['--preflight'] if cpu else [])],env)
os.execve=dispatch;sys.argv=[str(ROOT/'bootstrap.py')]
runpy.run_path(str(ROOT/'bootstrap.py'),run_name='__main__')
