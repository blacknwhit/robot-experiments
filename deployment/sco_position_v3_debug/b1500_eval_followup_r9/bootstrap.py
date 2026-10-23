from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
preflight='--cpu' in sys.argv;original=os.execve
def dispatch(python,argv,env):
    assert argv==[python,str(ROOT/'controller.py')]
    if preflight:env['CUDA_VISIBLE_DEVICES']=''
    original(python,[python,str(HERE/'controller.py'),*(['--preflight'] if preflight else [])],env)
os.execve=dispatch;sys.argv=[str(ROOT/'bootstrap.py')]
runpy.run_path(str(ROOT/'bootstrap.py'),run_name='__main__')
