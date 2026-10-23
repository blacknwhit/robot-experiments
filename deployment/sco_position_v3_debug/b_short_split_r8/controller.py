"""B1500 train-only continuation. Evaluation is validated separately on fslgpu."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,math,os,socket,subprocess,sys,time,traceback,uuid
import pyarrow  # keep the already validated import order
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO=ROOT/'repo';CONTROL=HERE/'control'
sys.path.insert(0,str(REPO/'scripts'))
from common import activate,environment,sha256
activate('embedding3d_temporal_b')
from checkpoint import choose_latest,validate
from contracts import check_recipe,select_training,validate_complete
import yaml
CFG=ROOT/'configs/embedding3d_temporal_b_short_1500.yaml'
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def verify(require_evaluation=True):
    assert sha256(ROOT/'release_manifest.json')=='65c003c6766692a299eed7f58d075bbc7060e1ed8c0338f8bfec4c198a3c0dbb'
    for directory,name in [(ROOT,'release_manifest.json'),(HERE,'manifest.json')]:
        for rel,digest in json.loads((directory/name).read_text())['files'].items():
            p=directory/rel;p.resolve().relative_to(ROOT);assert sha256(p)==digest,rel
    cfg=check_recipe(yaml.safe_load(CFG.read_text()))
    for variant in ('embedding3d_pos_a','embedding3d_temporal_b'):
        proof=json.loads((ROOT/'control'/f'{variant}_GPU_ACCEPTED.json').read_text())
        assert proof['passed'] and proof['release_sha256']==sha256(ROOT/'release_manifest.json')
    if require_evaluation:
        evaluation_root=ROOT/'fslgpu_acceptance_r7_20261023'
        assert sha256(evaluation_root/'manifest.json')=='cac4ae92d32f9ecded7d0304db6b4c3c28233294ce9eba4f55dc2ec955f606af'
        for rel,digest in json.loads((evaluation_root/'manifest.json').read_text())['files'].items():
            assert sha256(evaluation_root/rel)==digest,rel
        gate=ROOT/'fslgpu_acceptance_r7_20261023/control/FSLGPU_EVALUATION_ACCEPTED.json'
        proof=json.loads(gate.read_text());assert proof['passed']
        rows={r['case']:r for r in proof['records']}
        assert rows['a_default']['passed'] and rows['b_default']['passed']
        assert rows['a_default']['step']==1500 and rows['b_default']['step']==3
        assert rows['a_default']['variant']=='embedding3d_pos_a' and rows['b_default']['variant']=='embedding3d_temporal_b'
        assert rows['a_default']['strict_load']['all_loaded_values_exact_at_model_dtype']
        assert rows['b_default']['strict_load']['all_loaded_values_exact_at_model_dtype']
        # This gate permits split engineering validation, not clearance of the old ACP EGL failure.
        assert proof['ACP_bug_fixed'] is False
    return cfg
def main():
    if '--preflight' in sys.argv:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        verify(False)
        subprocess.run([sys.executable,str(REPO/'scripts/train.py'),'--config',str(CFG),'--dry-run'],check=True)
        subprocess.run([sys.executable,str(REPO/'scripts/import_preflight.py'),'--config',str(CFG)],env=environment('embedding3d_temporal_b'),check=True)
        print('B1500_CPU_PREFLIGHT_PASSED');return
    CONTROL.mkdir(exist_ok=True)
    with (ROOT/'control/pipeline.lock').open('r') as writer,(CONTROL/'writer.lock').open('a+') as owner:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'HOLD_FOR_MAIN.json').exists() or (CONTROL/'B_SHORT1500_COMPLETE.json').exists():return
        attempt={'attempt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
                 'hostname':socket.gethostname(),'started_unix':time.time(),'pid':os.getpid(),
                 'world_size':2,'per_device_batch':32,'global_batch':64,'accumulation':1,'max_short_steps':1500,'formal_training':False}
        directory=CONTROL/'attempts'/attempt['attempt'];directory.mkdir(parents=True);save(CONTROL/'current_attempt.json',attempt)
        def phase(value,**extra):
            data={**attempt,'phase':value,'time_unix':time.time(),**extra};save(CONTROL/'phase.json',data);print('POSITION_VALIDATION_PHASE '+json.dumps(data),flush=True)
        def run(command,name,env=None):
            with (directory/name).open('x') as log:
                proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
                try:
                    for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                    if proc.wait():raise RuntimeError('Subprocess failed: '+name)
                finally:
                    if proc.poll() is None:
                        spec=importlib.util.spec_from_file_location('cleanup',ROOT/'reference_eval_r6_20261023/gdb_runner.py')
                        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.stop_owned_tree(proc)
        try:
            cfg=verify();out=Path(cfg['run_root_dir'])/cfg['run_id'];out.resolve().relative_to(ROOT)
            attempt['release_sha256']=sha256(ROOT/'release_manifest.json');save(CONTROL/'current_attempt.json',attempt)
            for variant in ('embedding3d_pos_a','embedding3d_temporal_b'):
                save(CONTROL/f'{variant}_GPU_ACCEPTED.json',json.loads((ROOT/'control'/f'{variant}_GPU_ACCEPTED.json').read_text()))
            phase('hardware_preflight')
            run([sys.executable,str(REPO/'scripts/import_preflight.py'),'--config',str(CFG)],'imports.log',environment('embedding3d_temporal_b'))
            run([sys.executable,str(ROOT/'cuda_environment_probe.py'),'--output',str(directory/'cuda_environment.json')],'cuda_environment.log')
            import torch
            assert torch.cuda.device_count()==2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2))
            run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(REPO/'scripts/nccl_probe.py')],'nccl.log')
            claim=CONTROL/'B_SHORT_BASE_CLAIM.json'
            select_complete=lambda run,cfg:choose_latest(run,cfg,validator=lambda p,c:validate_complete(p,c,validate))
            selection=select_training(out,cfg,claim,select_complete);save(directory/'selection.json',selection)
            metrics=out/'point_track_metrics.jsonl';offset=metrics.stat().st_size if metrics.exists() else 0
            phase('B_short_training',run=str(out),metric_start_offset=offset,**selection)
            if selection['step']<1500:
                if not selection['resume']:
                    with claim.open('x') as f:json.dump(attempt,f,indent=2)
                cmd=[sys.executable,str(REPO/'scripts/train.py'),'--config',str(CFG)]
                if selection['resume']:cmd.append('--resume')
                run(cmd,'B_short_1500.log')
            assert validate(out/'checkpoints/checkpoint-1500',cfg)==1500
            for step in (1,1000,1500):assert (out/f'checkpoints/checkpoint-{step}/PORTABLE_COMPLETE.json').exists()
            rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            assert any(r['step']==1500 for r in rows)
            assert all(math.isfinite(r[k]) for r in rows for k in ('total_loss','action_dit_loss','position_loss','difference_loss','gradient_norm'))
            save(CONTROL/'B_SHORT1500_COMPLETE.json',{**attempt,'passed':True,'step':1500,'run':str(out),'evaluation_pending':True,'full_benchmark':False})
            phase('B_short1500_complete_evaluation_pending',run=str(out),metric_start_offset=offset)
        except Exception:
            save(CONTROL/'HOLD_FOR_MAIN.json',{**attempt,'error':traceback.format_exc()});phase('application_error_hold');traceback.print_exc()
if __name__=='__main__':main()
