"""Use immutable r4 evaluation with an explicit evaluation-only allocator policy."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('embedding3d_pos_a','embedding3d_temporal_b'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator',choices=('original','default'),default='default')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.relative_to(HERE);out.mkdir(parents=True,exist_ok=True)
    a.checkpoint.resolve().relative_to(ROOT);a.config.resolve().relative_to(ROOT)
    if a.allocator=='default':
        for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):os.environ.pop(key,None)
    audit=out/'runtime_audit';audit.mkdir(exist_ok=True);os.environ['EVAL_AUDIT_ATTEMPT']=str(audit)
    sys.path.insert(0,str(ROOT/'repo/scripts'))
    entry=ROOT/'eval_repair_r4_20261022/evaluate.py'
    spec=importlib.util.spec_from_file_location('immutable_evaluate',entry);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    argv=[str(entry),'--variant',a.variant,'--home',str(ROOT/'home'),'--config',str(a.config),'--checkpoint',str(a.checkpoint),
          '--sim-python',str(HERE/'sim_debug.sh'),'--sim-ld-preload','/lib/x86_64-linux-gnu/libffi.so.7',
          '--output',str(out),'--gpu-ids','0','--egl-device-ids','0','--port-base','11790','--smoke']
    if a.dry_run:argv.append('--dry-run')
    (out/'launch.json').write_text(json.dumps({'argv':argv,'allocator':a.allocator,
        'environment':{k:os.environ.get(k) for k in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')},'training_started':False},indent=2))
    sys.argv=argv;module.main()
if __name__=='__main__':main()
