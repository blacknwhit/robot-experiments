from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
mode='cpu_preflight.py' if '--cpu' in sys.argv else 'controller.py'
original=os.execve
def dispatch(python,argv,env):
    assert argv==[python,str(ROOT/'controller.py')]
    env['PYTHONFAULTHANDLER']='1'
    if mode=='cpu_preflight.py':env['CUDA_VISIBLE_DEVICES']=''
    original(python,[python,str(HERE/mode)],env)
os.execve=dispatch;sys.argv=[str(ROOT/'bootstrap.py')]
runpy.run_path(str(ROOT/'bootstrap.py'),run_name='__main__')
