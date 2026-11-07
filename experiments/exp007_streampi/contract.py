"""Recipe, source identity and path boundary; no ML imports."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def identity(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

def recipe_identity(cfg):
    return identity({k:v for k,v in cfg.items() if k not in ('run_root','mode','gpu_acceptance')})

def validate(cfg):
    assert cfg['gpus']==8 and cfg['global_batch_size']==64
    assert cfg['per_device_batch_size']==8 and cfg['gradient_accumulation_steps']==1
    assert cfg['gpus']*cfg['per_device_batch_size']*cfg['gradient_accumulation_steps']==cfg['global_batch_size']
    assert cfg['fsdp_devices']==8 and cfg['optimizer_execution']=='gpu_fsdp'
    assert cfg['num_train_steps']==70000
    assert cfg['history_frames']==3 and cfg['history_interval']==5 and not cfg['enable_jitter']
    assert cfg['training_anchor']=='all_episode_frames' and cfg['discrete_state_input']
    assert cfg['action_horizon']==50 and cfg['action_dim']==32 and cfg['environment_action_dim']==12
    assert cfg['learning_rate']=={'warmup_steps':10000,'peak_lr':5e-5,'decay_steps':1000000,'decay_lr':5e-5}
    assert cfg['ema_decay']==.999 and cfg['seed']==42
    assert cfg['save_interval']>0 and cfg['mode'] in ('smoke','formal')
    assert cfg['source_commit']=='ba024b2149ed94542a6c3352554f4f52f766a26d'

def source_manifest():
    paths=list(ROOT.glob('*.py'))+list((REPO/'runtimes/streampi/src').rglob('*.py'))
    paths+=list((REPO/'runtimes/streampi_reader').rglob('*.py'))
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}

def load(path,*,verify_sources=True):
    cfg=json.loads(Path(path).read_text());validate(cfg)
    if verify_sources:
        current=source_manifest()
        if current!=cfg['source_files']:raise ValueError('Code changed after configure; do not resume under changed code')
    scope=Path(cfg['workspace_root']).resolve()
    Path(cfg['run_root']).resolve().relative_to(scope)
    for key in ('base_checkpoint','norm_assets','tokenizer'):
        Path(cfg[key]).resolve().relative_to(scope)
    for key in ('data_root_dir','official_registry_file','v3_arrow_cache_dir'):
        Path(cfg['reader'][key]).resolve().relative_to(scope)
    if cfg['mode']=='formal':
        gate=cfg.get('gpu_acceptance',{})
        if not gate.get('passed') or gate.get('physical_gpu_count')!=8 or gate.get('recipe_identity')!=recipe_identity(cfg):
            raise ValueError('Formal run requires exact-recipe actual eight-GPU save/restart acceptance')
    return cfg

def verify_assets(cfg):
    for path,record in cfg['resource_manifest'].items():
        p=Path(path)
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise ValueError(f'Resource fingerprint changed: {p}')
