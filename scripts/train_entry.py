"""Worker entry: preserves scientific trainers; adds portability and save markers."""
import pyarrow  # Import before Torch/JAX, matching the accepted runtime order.
import argparse, copy, json, os, time
from pathlib import Path
import yaml
from common import activate, is_point_variant, POSITION_VARIANTS
from release_contract import verify_release

def canonical(c):
    from checkpoint import canonical_recipe
    return canonical_recipe(c)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());verify_release(c)
    v=c['portable']['variant'];activate(v)
    if v == 'pri4r_paper':
        from pri4r_contract import validate_recipe
        validate_recipe(c)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if v in POSITION_VARIANTS:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        modes = [bool(c['portable'].get(k)) for k in ('smoke','engineering','formal')]
        limit = 10 if modes[0] else (1500 if modes[1] else 70000)
        if sum(modes) != 1 or not 1 <= c['trainer']['stop_after_steps'] <= limit:
            raise ValueError('V3 requires exactly one explicit bounded or formal candidate mode')
        if modes[2] and (c['experiment'].get('phase') != 'formal_candidate'
                         or c['experiment'].get('formal_training_authorized') is not True):
            raise ValueError('Generate a formal candidate recipe with configure.py --formal')
    import torch
    import torch.distributed as dist
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK'])
    batch=128 if v=='pri4r_paper' else 64
    if world!=c['portable']['world_size'] or c['datasets']['vla_data']['per_device_batch_size']*world*c['trainer']['gradient_accumulation_steps']!=batch:raise ValueError('World/batch mismatch')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    run=Path(c['run_root_dir'])/c['run_id'];receipt=run/'portable_recipe.json'
    from checkpoint import choose_latest, validate
    if a.resume:
        if not receipt.is_file() or canonical(json.loads(receipt.read_text()))!=canonical(c):raise ValueError('Resume recipe/code differs')
        selection=[None,None]
        if rank==0:
            try:
                path,rejected=choose_latest(run,c);selection[0]=str(path)
                print('FULL_STATE_RESUME',path,'REJECTED',rejected,flush=True)
            except Exception as e:selection[1]=repr(e)
        dist.broadcast_object_list(selection,src=0)
        if selection[1]:raise RuntimeError(selection[1])
        c['trainer'].update(is_resume=True,resume_from_checkpoint=selection[0])
    else:
        if rank==0 and (receipt.exists() or (run/'config.full.yaml').exists()):raise FileExistsError('Existing run requires --resume or a new run-id')
    for path in (c['framework']['pi05']['init_checkpoint'],str(Path(c['framework']['pi05']['assets_dir'])/'norm_stats.json'),c['datasets']['vla_data']['official_registry_file']):
        if not Path(path).is_file():raise FileNotFoundError(path)
    if is_point_variant(v):
        gt=Path(c['datasets']['vla_data']['point_track_gt_root'])
        accepted=Path(c['portable'].get('gt_acceptance_receipt', str(gt/'PORTABLE_GT_ACCEPTED.json')))
        if not accepted.is_file():raise RuntimeError('Run scripts/prepare_gt.py --stage verify first')
        proof=json.loads(accepted.read_text())
        if v=='pri4r_paper':
            from pri4r_contract import validate_gt_receipt
            validate_gt_receipt(c,proof,gt)
        elif proof['episodes']!=30000 or proof['frames']!=27815381 or proof['config_sha256']!=c['datasets']['vla_data']['point_track_gt_config_sha256']:raise ValueError('GT scope/config differs')
    if rank==0:
        run.mkdir(parents=True,exist_ok=True)
        if not a.resume:receipt.write_text(json.dumps(c,indent=2))
        print('HARDWARE',json.dumps({'world':world,'GPUs':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'torch':str(torch.__version__)}),flush=True)
    from starVLA.training import train_starvla as native
    # An extra full first-update save and post-save audit marker do not alter updates.
    klass=native.VLATrainer
    if v=='query_fixed':
        from starVLA.training.foresight_trainer import ForesightTrainer
        klass=ForesightTrainer
    elif v in POSITION_VARIANTS:
        from starVLA.training.position_point_trainer import PositionPointTrainer
        klass=PositionPointTrainer
    elif is_point_variant(v):
        from starVLA.training.query_point_trainer import QueryPointTrainer
        klass=QueryPointTrainer
    save=klass._save_checkpoint;log=klass._log_metrics
    def checked_save(self):
        started=time.perf_counter()
        save(self)
        if self.accelerator.is_main_process:
            step=self.completed_steps;cp=run/'checkpoints'/f'checkpoint-{step}'
            if v=='pri4r_paper' and c['framework']['pi05'].get('official_precision_mode'):
                from pri4r_native_state import write_metadata
                write_metadata(self,cp)
            validate(cp,c)
            (cp/'PORTABLE_COMPLETE.json').write_text(json.dumps({'step':step,'world':world,'recipe':canonical(c)},indent=2))
            with (run/'save_timing.jsonl').open('a') as f:
                f.write(json.dumps({'step':step,'save_and_validate_seconds':time.perf_counter()-started,
                    'checkpoint_bytes':sum(p.stat().st_size for p in cp.rglob('*') if p.is_file()),
                    'gpu_lifetime_peak_gb':torch.cuda.max_memory_allocated()/1e9})+'\n')
        self.accelerator.wait_for_everyone()
    def first_save(self,metrics):
        log(self,metrics)
        if self.completed_steps in (1,self.config.trainer.stop_after_steps) and self.completed_steps%self.config.trainer.save_interval!=0:self._save_checkpoint()
    klass._save_checkpoint=checked_save;klass._log_metrics=first_save
    if (v in POSITION_VARIANTS or v=='pri4r_paper') and c['portable'].get('smoke'):
        from engineering_audit import install
        install(klass)
    if v=='baseline':
        prepare=klass.prepare_training
        def audited_prepare(self):
            prepare(self)
            if self.resume_from_checkpoint:
                from starVLA.training.foresight_resume_audit import verify_loaded_partition
                verify_loaded_partition(self.model,self.resume_from_checkpoint,self.config.output_dir)
        klass.prepare_training=audited_prepare
    from omegaconf import OmegaConf
    cfg=native.apply_config_compat(OmegaConf.create(c));cfg.config_yaml=str(a.config)
    native.main(cfg)
if __name__=='__main__':main()
