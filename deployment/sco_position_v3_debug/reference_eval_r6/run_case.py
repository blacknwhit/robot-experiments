"""Same A1500 r4 entry with one allocator switch, plus the accepted baseline entry."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
from allocator_environment import CASES,case_environment,ALLOCATOR_KEYS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REFERENCE=ROOT.parent/'robot_experiments_eval_affinity_gpu_20261021'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args();out=a.output.resolve();out.relative_to(HERE);out.mkdir(parents=True,exist_ok=True)
    env=case_environment(os.environ,a.case);os.environ.clear();os.environ.update(env)
    (out/'runtime_audit').mkdir(exist_ok=True);os.environ['EVAL_AUDIT_ATTEMPT']=str(out/'runtime_audit')
    reference=a.case=='baseline_reference'
    repo=(REFERENCE if reference else ROOT)/'repo'
    sys.path.insert(0,str(repo/'scripts'))
    entry=repo/'scripts/evaluate.py' if reference else ROOT/'eval_repair_r4_20261022/evaluate.py'
    spec=importlib.util.spec_from_file_location('case_evaluate',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if reference:
        config=REFERENCE/'baseline.yaml';prepared=json.loads((REFERENCE/'prepared.json').read_text())
        checkpoint=Path(prepared['checkpoint']);variant='baseline';home=REFERENCE/'home'
    else:
        config=ROOT/'configs/embedding3d_pos_a_short_1500.yaml'
        checkpoint=ROOT/'home/engineering_runs/embedding3d_pos_a_short_r2/checkpoints/steps_1500_model.safetensors'
        variant='embedding3d_pos_a';home=ROOT/'home'
    argv=[str(entry),'--variant',variant,'--home',str(home),'--config',str(config),'--checkpoint',str(checkpoint),
          '--sim-python',str(HERE/'sim_debug.sh'),'--output',str(out),'--gpu-ids','0','--egl-device-ids','0',
          '--port-base','11590','--smoke']
    if not reference:argv+=['--sim-ld-preload','/lib/x86_64-linux-gnu/libffi.so.7']
    launch={'case':a.case,'argv':argv,'entry':str(entry),'allocator':{k:os.environ.get(k) for k in ALLOCATOR_KEYS},
            'capabilities':os.environ.get('NVIDIA_DRIVER_CAPABILITIES'),'training_started':False}
    (out/'launch.json').write_text(json.dumps(launch,indent=2))
    if a.dry_run:argv.append('--dry-run')
    sys.argv=argv;module.main()
if __name__=='__main__':main()
