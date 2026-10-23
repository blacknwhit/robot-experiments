"""Only the already authorized B1500 engineering recipe is accepted."""
def validate_complete(path,cfg,deep_validate):
    if not (path/'PORTABLE_COMPLETE.json').is_file():
        raise ValueError('Checkpoint lacks its completed-save marker')
    return deep_validate(path,cfg)

def check_recipe(cfg):
    assert cfg['portable']['variant']=='embedding3d_temporal_b'
    assert cfg['portable']['world_size']==2 and cfg['portable']['engineering'] is True
    assert cfg['datasets']['vla_data']['per_device_batch_size']==32
    t=cfg['trainer']
    assert t['gradient_accumulation_steps']==1 and t['expected_global_batch_size']==64
    assert t['stop_after_steps']==1500 and t['max_train_steps']==100000
    assert t['save_first_update'] is True and t['save_interval']==1000
    assert cfg['experiment']['formal_training_authorized'] is False
    assert cfg['experiment']['short_validation_max_steps']==1500
    assert cfg['run_id']=='embedding3d_temporal_b_short_r2'
    return cfg

def select_training(run,cfg,claim,choose_latest):
    checkpoints=run/'checkpoints'
    has_state=checkpoints.exists() and any(checkpoints.iterdir())
    if (run/'portable_recipe.json').exists() or has_state:
        selected,rejected=choose_latest(run,cfg)
        return {'resume':True,'checkpoint':str(selected),'step':int(selected.name.split('-')[-1]),'rejected':rejected}
    if claim.exists():
        raise RuntimeError('Base was already claimed but no complete state is available; refuse silent restart')
    return {'resume':False,'checkpoint':None,'step':0,'rejected':[]}
