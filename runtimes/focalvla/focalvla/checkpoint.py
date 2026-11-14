"""Atomic checkpoints with exact optimizer/RNG/data/scheduler restoration."""
from pathlib import Path
import random
import numpy as np
import torch
from .common import atomic_json, read_json, digest


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed(value)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'])


def next_batch_digest(draws, step):
    return digest(draws[step].tolist()) if step < len(draws) else 'finished'


def latest(root, contract):
    root = Path(root)
    candidates = []
    incompatible = []
    for marker in root.glob('step_*.json'):
        info = read_json(marker)
        file = root/info['file']
        if not file.exists() or file.stat().st_size != info['bytes']:
            continue  # Interrupted writes are never complete.
        if info['contract'] != contract:
            incompatible.append(marker.name)
        else:
            candidates.append((info['step'], file))
    if incompatible:
        raise ValueError(f'Incompatible checkpoints in output directory: {incompatible}')
    return max(candidates, default=(0, None))


def save(root, model, optimizer, step, contract, rank_rng, draws, global_batch, next_lr):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    file = root/f'step_{step:06d}.pt'
    tmp = file.with_suffix('.tmp')
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'contract': contract, 'rank_rng': rank_rng, 'scheduler': {'completed_steps': step, 'next_lr': next_lr},
             'data_progress': {'draws': step*global_batch, 'next_batch': next_batch_digest(draws, step)}}
    torch.save(state, tmp)
    # Flush to persistent storage before publishing the completeness marker.
    import os
    with tmp.open('rb') as stream: os.fsync(stream.fileno())
    tmp.replace(file)
    atomic_json(file.with_suffix('.json'), {'step': step, 'file': file.name, 'bytes': file.stat().st_size,
                                          'contract': contract, 'world_size': len(rank_rng)})
    return file


def restore(file, model, optimizer, contract, rank, world, draws, global_batch, lr_function):
    state = torch.load(file, map_location='cpu', weights_only=False, mmap=True)
    step = state['step']
    assert state['contract'] == contract and len(state['rank_rng']) == world
    assert state['data_progress'] == {'draws': step*global_batch, 'next_batch': next_batch_digest(draws, step)}
    assert state['scheduler'] == {'completed_steps': step, 'next_lr': lr_function(step)}
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    assert all(int(s['step']) == step for s in optimizer.state.values() if 'step' in s)
    restore_rng(state['rank_rng'][rank])
    return step

