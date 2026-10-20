"""Checkpointed v2 recipe identity; runtime paths/stop gates are not semantics."""
from pathlib import Path
import hashlib,json


def semantic_recipe(config):
    from omegaconf import OmegaConf
    if hasattr(config,'unwrap'):
        config=config.unwrap()
    c=OmegaConf.to_container(config,resolve=True) if OmegaConf.is_config(config) else config
    data=dict(c['datasets']['vla_data'])
    for k in ['num_workers','pin_memory','prefetch_factor','point_track_cache_size']:
        data.pop(k,None)
    t=c['trainer']
    keys=['max_train_steps','num_warmup_steps','learning_rate','lr_scheduler_type','scheduler_specific_kwargs',
          'optimizer','gradient_clipping','gradient_accumulation_steps','expected_global_batch_size','freeze_modules','ema_decay']
    return dict(schema='point_trajectory_training_v3',seed=c['seed'],framework=c['framework'],dataset=data,
                trainer={k:t.get(k) for k in keys},release_sha256=c['trajectory_release_sha256'])


def recipe_sha(config):
    return hashlib.sha256(json.dumps(semantic_recipe(config),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_checkpoint_contract(config,step):
    path=Path(config.output_dir)/'checkpoints'/f'checkpoint-{step}'/'trajectory_contract.json'
    value=dict(schema='point_trajectory_training_v3',completed_steps=int(step),semantic_recipe=semantic_recipe(config),recipe_sha256=recipe_sha(config))
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def validate_contract(path,config,step):
    record=json.loads(Path(path).read_text())
    if record.get('schema')!='point_trajectory_training_v3' or record['completed_steps']!=step or record['recipe_sha256']!=recipe_sha(config) or record['semantic_recipe']!=semantic_recipe(config):
        raise ValueError('Checkpoint objective/data/optimizer/release contract is incompatible')
