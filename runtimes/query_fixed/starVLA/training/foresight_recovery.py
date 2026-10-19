"""Complete-state selection and bounded rolling retention for this experiment."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import torch
import yaml
from safetensors import safe_open

MILESTONES = {70000, 75000, 100000}
TEACHER_SHA = 'b2b43d91a70d44e4feb7de096a4fa46718fea942c6d4cccc3697372d222d0e6c'
RUNTIME_TRAINER_KEYS = {'is_resume', 'resume_from_checkpoint', 'logging_frequency', 'save_interval',
                        'stop_after_steps', 'auto_resume_control_dir'}

def recipe_hash(cfg):
    trainer = {k: v for k, v in cfg['trainer'].items() if k not in RUNTIME_TRAINER_KEYS}
    semantic = {'seed': cfg['seed'], 'framework': cfg['framework'], 'datasets': cfg['datasets'], 'trainer': trainer}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def summary_entries(run):
    path = Path(run) / 'summary.jsonl'
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A writer killed mid-line has not committed a checkpoint.
    return entries

def validate_checkpoint(checkpoint, expected_recipe):
    checkpoint = Path(checkpoint).resolve()
    match = re.fullmatch(r'checkpoint-(\d+)', checkpoint.name)
    if not match:
        raise ValueError('Not a full-state checkpoint directory')
    step = int(match[1]); run = checkpoint.parent.parent
    cfg = yaml.safe_load((run / 'config.full.yaml').read_text())
    assert recipe_hash(cfg) == expected_recipe, 'Training recipe mismatch'
    identity = json.loads((run / 'teacher_identity.json').read_text())
    assert identity['teacher_sha256'] == TEACHER_SHA, 'Teacher identity mismatch'
    completed = [entry for entry in summary_entries(run)
                 if entry.get('steps') == step and entry.get('consumed_batches') == step
                 and Path(entry.get('state_checkpoint', '')).resolve() == checkpoint]
    assert completed, 'Missing committed summary entry'
    weights = Path(completed[-1]['evaluation_weights']).resolve()
    assert weights.parent == checkpoint.parent and weights.name == f'steps_{step}_model.safetensors'
    required = [checkpoint / 'custom_checkpoint_0.pkl', checkpoint / 'custom_checkpoint_1.pkl',
                checkpoint / 'latest', checkpoint / 'pytorch_model/mp_rank_00_model_states.pt', weights]
    decay = cfg['framework']['pi05']['foresight'].get('teacher_ema_decay')
    if decay is not None:
        required.append(checkpoint / 'custom_checkpoint_2.pkl')
    required += [checkpoint / f'random_states_{rank}.pkl' for rank in range(4)]
    required += [checkpoint / f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(4)]
    assert all(p.is_file() and p.stat().st_size > 0 for p in required), 'Missing checkpoint shard'
    assert (checkpoint / 'latest').read_text().strip() == 'pytorch_model'
    for path in required:
        if path.suffix in {'.pt', '.pkl'}:
            with zipfile.ZipFile(path) as archive:
                assert archive.infolist(), f'Incomplete ZIP archive: {path}'
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False, mmap=True)
    progress = load(checkpoint / 'custom_checkpoint_1.pkl')
    scheduler = load(checkpoint / 'custom_checkpoint_0.pkl')
    assert progress['completed_steps'] == progress['consumed_batches'] == step
    assert scheduler['last_epoch'] == step
    model = load(checkpoint / 'pytorch_model/mp_rank_00_model_states.pt')
    ds = model['ds_config']
    assert model['dp_world_size'] == 4 and model['mp_world_size'] == 1
    assert model['global_steps'] == step and model['global_samples'] == step * 64
    assert ds['train_batch_size'] == 64 and ds['train_micro_batch_size_per_gpu'] == 16
    assert ds['gradient_accumulation_steps'] == 1 and ds['zero_optimization']['stage'] == 2
    sizes = [sum(shape.numel() for shape in group.values()) // 4 for group in model['param_shapes']]
    assert tuple(model['module']['policy_model.future_queries'].shape) == (1, 64, 2048)
    if decay is not None:
        from openpi.models_pytorch.foresight_ema import validate_ema_state
        prefix = 'policy_model.paligemma_with_expert.paligemma.model.vision_tower.'
        reference = {k[len(prefix):]: v for k,v in model['module'].items() if k.startswith(prefix)}
        validate_ema_state(load(checkpoint / 'custom_checkpoint_2.pkl'), decay=decay,
                           identity=identity, reference=reference, updates=step)
    del model
    for rank in range(4):
        opt = load(checkpoint / f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')['optimizer_state_dict']
        assert opt['partition_count'] == [4] * len(sizes)
        for i, size in enumerate(sizes):
            state = opt['base_optimizer_state']['state'][i]
            assert state['step'].item() == step
            assert opt['base_optimizer_state']['param_groups'][i]['lr'] == scheduler['_last_lr'][i]
            for tensor in [opt['single_partition_of_fp32_groups'][i], state['exp_avg'], state['exp_avg_sq']]:
                assert tensor.numel() == size
                assert bool(torch.isfinite(tensor[:16]).all() and torch.isfinite(tensor[-16:]).all())
        del opt
        rng = load(checkpoint / f'random_states_{rank}.pkl')
        assert {'random_state', 'numpy_random_seed', 'torch_manual_seed', 'torch_cuda_manual_seed'} <= rng.keys()
        assert len(rng['torch_cuda_manual_seed']) == 4
    with safe_open(weights, framework='pt', device='cpu') as f:
        query = f.get_tensor('policy_model.future_queries')
        assert tuple(query.shape) == (1, 64, 2048) and bool(torch.isfinite(query).all())
    return {'step': step, 'checkpoint': str(checkpoint), 'weights': str(weights), 'run': str(run),
            'recipe_sha256': expected_recipe, 'world_size': 4, 'global_batch_size': 64,
            'files': [{'path': str(p), 'bytes': p.stat().st_size} for p in required]}

def choose_checkpoint(runs, expected_recipe):
    candidates = []
    for run in runs:
        for checkpoint in (Path(run) / 'checkpoints').glob('checkpoint-*'):
            if re.fullmatch(r'checkpoint-\d+', checkpoint.name):
                candidates.append(checkpoint)
    candidates.sort(key=lambda p: (int(p.name.split('-')[-1]), p.stat().st_mtime_ns), reverse=True)
    rejected = []
    for checkpoint in candidates:
        try:
            return validate_checkpoint(checkpoint, expected_recipe), rejected
        except Exception as exc:
            rejected.append({'checkpoint': str(checkpoint), 'reason': str(exc)})
    raise RuntimeError(f'No complete matching checkpoint available: {rejected}')

def managed_run(run, control, expected_recipe):
    run = Path(run)
    assert not run.is_symlink()
    assert run.resolve().parent == (Path(control).resolve() / 'attempts')
    assert re.fullmatch(r'attempt_[A-Za-z0-9_]+', run.name)
    manifest = json.loads((run / 'attempt.json').read_text())
    assert manifest['recipe_sha256'] == expected_recipe
    return run.resolve()

def prune_managed(control, newest, expected_recipe):
    """Only remove this controller's artifacts after a new complete state is committed."""
    control = Path(control).resolve()
    newest_path = Path(newest['checkpoint']).resolve()
    assert newest_path.is_relative_to(control / 'attempts')
    assert (newest_path / 'foresight_complete.json').is_file()
    keep = {newest_path}
    runs = []
    milestones = {}
    for candidate in sorted((control / 'attempts').glob('attempt_*')):
        run = managed_run(candidate, control, expected_recipe)
        runs.append(run)
        for cp in (run / 'checkpoints').glob('checkpoint-*'):
            if re.fullmatch(r'checkpoint-\d+', cp.name) and int(cp.name.split('-')[-1]) in MILESTONES:
                marker = cp / 'foresight_complete.json'
                if not marker.exists():
                    # A kill after the trainer's summary append but before our marker
                    # must not cause a valid permanent milestone to be discarded.
                    try:
                        recovered = validate_checkpoint(cp, expected_recipe)
                    except Exception:
                        continue
                    atomic_json(marker, recovered)
                if marker.exists():
                    record = json.loads(marker.read_text())
                    assert record['recipe_sha256'] == expected_recipe
                    assert record['step'] == int(cp.name.split('-')[-1])
                    previous = milestones.get(record['step'])
                    if previous is None or marker.stat().st_mtime_ns > (previous / 'foresight_complete.json').stat().st_mtime_ns:
                        milestones[record['step']] = cp.resolve()
    keep.update(milestones.values())
    removed = []
    for run in runs:
        parent = run / 'checkpoints'
        for cp in sorted(parent.glob('checkpoint-*')):
            if not re.fullmatch(r'checkpoint-\d+', cp.name):
                continue
            target = cp.resolve()
            assert not cp.is_symlink() and target.parent == parent.resolve()
            if target not in keep:
                shutil.rmtree(target)
                removed.append(str(target))
        kept_steps = {int(cp.name.split('-')[-1]) for cp in keep if cp.parent == parent}
        for weights in parent.glob('steps_*_model.safetensors'):
            match = re.fullmatch(r'steps_(\d+)_model.safetensors', weights.name)
            if match and int(match[1]) not in kept_steps:
                target = weights.resolve()
                assert not weights.is_symlink() and target.parent == parent.resolve()
                target.unlink()
                removed.append(str(target))
    with (control / 'retention.jsonl').open('a') as f:
        f.write(json.dumps({'latest_step': newest['step'], 'kept_milestones': sorted(MILESTONES), 'removed': removed}) + '\n')
    return removed

def commit_and_prune(run, step, control):
    cfg = yaml.safe_load((Path(run) / 'config.full.yaml').read_text())
    signature = recipe_hash(cfg)
    managed_run(run, control, signature)
    record = validate_checkpoint(Path(run) / 'checkpoints' / f'checkpoint-{step}', signature)
    atomic_json(Path(record['checkpoint']) / 'foresight_complete.json', record)
    atomic_json(Path(control) / 'latest_complete.json', record)
    removed = prune_managed(control, record, signature)
    print('FORESIGHT_ROLLING_CHECKPOINT_COMMITTED', step, 'removed_artifacts', len(removed), flush=True)
    return record
