"""Worker entry: preserves scientific trainers; adds portability and save markers."""
import pyarrow  # Import before Torch/JAX, matching the accepted runtime order.
import argparse, copy, json, os
from pathlib import Path
import yaml
from common import activate, is_point_variant, POSITION_VARIANTS

def canonical(c):
    c=copy.deepcopy(c)
    for k in ('config_yaml','output_dir'):c.pop(k,None)
    for k in ('is_resume','resume_from_checkpoint','stop_after_steps'):c['trainer'].pop(k,None)
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());v=c['portable']['variant'];activate(v)
    if v in POSITION_VARIANTS:
        limit = 10 if c['portable'].get('smoke') else 1500
        if (not (c['portable'].get('smoke') or c['portable'].get('engineering'))
                or not 1 <= c['trainer']['stop_after_steps'] <= limit):
            raise ValueError('V3 formal training has not been authorized/calibrated')
    import torch
    import torch.distributed as dist
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK'])
    if world!=c['portable']['world_size'] or c['datasets']['vla_data']['per_device_batch_size']*world!=64:raise ValueError('World/batch mismatch')
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
        if proof['episodes']!=30000 or proof['frames']!=27815381 or proof['config_sha256']!=c['datasets']['vla_data']['point_track_gt_config_sha256']:raise ValueError('GT scope/config differs')
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
        save(self)
        if self.accelerator.is_main_process:
            step=self.completed_steps;cp=run/'checkpoints'/f'checkpoint-{step}'
            validate(cp,c)
            (cp/'PORTABLE_COMPLETE.json').write_text(json.dumps({'step':step,'world':world,'recipe':canonical(c)},indent=2))
        self.accelerator.wait_for_everyone()
    def first_save(self,metrics):
        log(self,metrics)
        if self.completed_steps in (1,self.config.trainer.stop_after_steps) and self.completed_steps%self.config.trainer.save_interval!=0:self._save_checkpoint()
    klass._save_checkpoint=checked_save;klass._log_metrics=first_save
    if v in POSITION_VARIANTS and c['portable'].get('smoke'):
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
