from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
mode='validate_on_fslgpu.py' if '--fslgpu' in sys.argv else 'controller.py'
original=os.execve
def dispatch(python,argv,env):
    assert argv==[python,str(ROOT/'controller.py')]
    env['PYTHONFAULTHANDLER']='1'
    original(python,[python,str(HERE/mode)],env)
os.execve=dispatch;sys.argv=[str(ROOT/'bootstrap.py')]
runpy.run_path(str(ROOT/'bootstrap.py'),run_name='__main__')
