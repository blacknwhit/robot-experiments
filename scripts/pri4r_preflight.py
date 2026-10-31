"""CPU-only resource/recipe check or complete-state check for the 8GPU70k run."""
import argparse,json
from pathlib import Path
import yaml
from common import sha256
from checkpoint import choose_latest
from pri4r_contract import validate_recipe,validate_gt_receipt
from pri4r_profiles import PROFILE_8GPU70K
from release_contract import verify_release

BASE_SHA='9fd36e337832960bbfc122bd686dfc64b89551522d891c0d21ae586385de297e'
NORM_SHA='4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'


def check_profile(config):
    validate_recipe(config)
    if config['portable'].get('training_profile')!=PROFILE_8GPU70K:
        raise ValueError('This entry requires the explicit 8GPU70k training profile')


def complete_checkpoint(config, selector=choose_latest):
    check_profile(config)
    run=Path(config['run_root_dir'])/config['run_id']
    checkpoint,rejected=selector(run,config)
    if checkpoint.name!='checkpoint-70000':
        raise ValueError('Training has not completed a compatible full checkpoint-70000; latest='+checkpoint.name)
    return {'passed':True,'step':70000,'checkpoint':str(checkpoint),
        'ema_model':str(checkpoint.parent/'steps_70000_model.safetensors'),
        'raw_model':str(checkpoint.parent/'steps_70000_raw_model.safetensors'),
        'rejected_checkpoints':rejected,'full_state_checked':True}


def check_resources(config):
    check_profile(config)
    d=config['datasets']['vla_data'];home=Path(config['portable']['resource_home'])
    gt=Path(d['point_track_gt_root'])
    proof=json.loads((gt/'PORTABLE_GT_ACCEPTED.json').read_text())
    validate_gt_receipt(config,proof,gt)
    scope=json.loads((home/'data/training_scope.json').read_text())
    if (scope['tasks'],scope['episodes'],scope['frames'])!=(300,30000,27815381):
        raise ValueError('Training scope differs from the pi05 baseline')
    if scope['registry_sha256']!=sha256(d['official_registry_file']):
        raise ValueError('Training registry changed')
    p=config['framework']['pi05']
    for path,expected in ((Path(p['init_checkpoint']),BASE_SHA),(Path(p['assets_dir'])/'norm_stats.json',NORM_SHA)):
        if sha256(path)!=expected:raise ValueError('Wrong original base/norm asset: '+str(path))
    for path in (home/'cache/openpi/big_vision/paligemma_tokenizer.model',home/'cache/arrow'):
        if not path.exists():raise FileNotFoundError(path)
    return {'passed':True,'training_profile':PROFILE_8GPU70K,'world':8,'micro_batch':16,
        'accumulation':1,'global_batch':128,'max_train_steps':70000,'stop_after_steps':70000,
        'warmup_steps':10000,'cosine_decay_steps':70000,'save_interval':5000,
        'gt_episodes':proof['episodes'],'gt_frames':proof['frames'],'model_allocated':False,'gpu_test':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--completed',action='store_true',help='Require full70k state and valid EMA export; never trains')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    verify_release(config)
    result=complete_checkpoint(config) if args.completed else check_resources(config)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
