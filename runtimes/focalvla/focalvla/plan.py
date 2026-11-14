"""Deterministic full-scope sampling and deduplicated offline work list."""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import atomic_json, read_json, digest, sha256


def build(config, resources, output):
    import pyarrow.parquet as pq
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scope = read_json(resources['training_scope'])['selected_episodes']
    counts = Counter(s['task_name'] for s in scope)
    assert len(scope) == config['expected_episodes'], len(scope)
    assert len(counts) == config['expected_tasks'], len(counts)
    assert set(counts.values()) == {config['episodes_per_task']}, counts
    assert len({(s['dataset_name'], s['episode_index']) for s in scope}) == len(scope)
    lookup = {}
    for ds in sorted({s['dataset_name'] for s in scope}):
        for row in pq.read_table(Path(resources['dataset_root']) / ds / 'meta/episodes').to_pylist():
            lookup[ds, row['episode_index']] = row
    episodes = []
    for item in sorted(scope, key=lambda s: (s['task_name'], s['dataset_name'], s['episode_index'])):
        row = dict(lookup[item['dataset_name'], item['episode_index']])
        row.update({k: item[k] for k in ('task_name', 'dataset_name')})
        row['id'] = len(episodes)
        row['name'] = f"{row['task_name']}__{row['source_episode_index']:06d}"
        # Keep short demonstrations and tail frames. Terminal observations/actions
        # are extended with their last recorded value; never cross episode boundaries.
        row['eligible_frames'] = row['length']
        assert row['eligible_frames'] > 0, row['name']
        episodes.append(row)
    assert len({r['name'] for r in episodes}) == len(episodes)
    identity = {'scope_sha256': sha256(resources['training_scope']), 'episodes': episodes,
                'seed': config['seed'], 'total_steps': config['total_steps'],
                'global_batch': config['global_batch'], 'future_offsets': config['future_offsets'],
                'task_sampling_alpha': config['task_sampling_alpha'], 'terminal_policy':'repeat_last'}
    contract = digest(identity)
    marker = output / 'PLAN.json'
    if marker.exists():
        old = read_json(marker)
        if old['contract'] != contract:
            raise ValueError('Prepared directory has a different sampling plan. Use a new directory.')
        return old
    n = config['global_batch'] * config['total_steps']
    rng = np.random.default_rng(config['seed'])
    tasks = sorted(counts)
    groups = [[r for r in episodes if r['task_name'] == task] for task in tasks]
    lengths = np.array([sum(r['eligible_frames'] for r in g) for g in groups])
    p = lengths.astype(np.float64) ** config['task_sampling_alpha']; p /= p.sum()
    selected = rng.choice(len(tasks), size=n, p=p)
    packed = np.empty(n, dtype=np.uint64)
    for task_id, group in enumerate(groups):
        positions = np.flatnonzero(selected == task_id)
        cumulative = np.cumsum([r['eligible_frames'] for r in group])
        offsets = rng.integers(0, int(cumulative[-1]), size=len(positions))
        local_episode = np.searchsorted(cumulative, offsets, side='right')
        starts = np.concatenate([[0], cumulative[:-1]])
        frame = offsets - starts[local_episode]
        episode_id = np.array([r['id'] for r in group])[local_episode]
        packed[positions] = (episode_id.astype(np.uint64) << np.uint64(32)) | frame.astype(np.uint64)
    unique, inverse, repeats = np.unique(packed, return_inverse=True, return_counts=True)
    episode_id = (unique >> np.uint64(32)).astype(np.int32)
    frame = (unique & np.uint64(0xffffffff)).astype(np.int32)
    first = np.searchsorted(episode_id, np.arange(len(episodes)))
    within = np.arange(len(unique)) - first[episode_id]
    records = np.stack([episode_id, frame, within], 1).astype(np.int32)
    np.save(output/'records.npy', records)
    np.save(output/'draws.npy', inverse.reshape(config['total_steps'], config['global_batch']).astype(np.int32))
    np.save(output/'repeats.npy', repeats.astype(np.int32))
    atomic_json(output/'episodes.json', episodes)
    result = {'contract': contract, 'episodes': len(episodes), 'tasks': len(tasks),
              'draws': n, 'unique_frames': len(unique), 'seed': config['seed'],
              'sampling': 'task probability proportional to eligible frames ** alpha; uniform frames within task',
              'task_sampling_alpha': config['task_sampling_alpha'], 'future_offsets': config['future_offsets'],
              'terminal_policy':'repeat_last', 'short_episodes':sum(r['length']<=50 for r in episodes),
              'scope_sha256': identity['scope_sha256'],
              'files': {f: sha256(output/f) for f in ('records.npy','draws.npy','repeats.npy','episodes.json')}}
    atomic_json(marker, result)
    return result

