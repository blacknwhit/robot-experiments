"""Final deployed combination: manifest, real child environment, three CLI dry runs, GDB."""
from pathlib import Path
import json,os,subprocess,sys,time,unittest
from allocator_environment import CASES,ALLOCATOR_KEYS,case_environment
from controller import verify
from gdb_runner import prerequisites
HERE=Path(__file__).resolve().parent
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    verify()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_allocator_environment.py')
    result=unittest.TextTestRunner().run(suite);assert result.wasSuccessful()
    directory=HERE/('cpu_'+str(time.time_ns()));directory.mkdir()
    for case in CASES:
        parent=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTORCH_ALLOC_CONF='expandable_segments:True')
        env=case_environment(parent,case)
        code='import json,os;print(json.dumps({k:os.environ.get(k) for k in '+repr(ALLOCATOR_KEYS)+'}))'
        observed=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
        assert observed=={k:env.get(k) for k in ALLOCATOR_KEYS}
        subprocess.run([sys.executable,str(HERE/'run_case.py'),'--case',case,'--output',str(directory/case),'--dry-run'],env=env,check=True)
    prerequisites()
    (HERE/'CPU_PREFLIGHT.json').write_text(json.dumps({'passed':True,'directory':str(directory),'tests':result.testsRun,'GPU_passed':False},indent=2))
if __name__=='__main__':main()
