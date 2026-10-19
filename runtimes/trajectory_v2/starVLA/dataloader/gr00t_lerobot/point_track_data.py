"""Attach offline point labels without altering the original Query sample selection."""
import re

from starVLA.pri4r.reader import PointTrackStore


def read_point_track(dataset, episode, current):
    cfg = dataset.data_cfg
    expected = str(cfg.get('point_track_gt_config_sha256', ''))
    if re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError('An accepted, pinned point GT configuration SHA256 is required')
    metadata = dataset.trajectory_ids_to_metadata[episode]
    if not {'source_prefix', 'source_episode_index'} <= metadata.keys():
        raise ValueError('Missing source episode mapping; aggregate episode IDs are not source IDs')
    if not hasattr(dataset, '_point_track_store'):
        dataset._point_track_store = PointTrackStore(
            cfg.point_track_gt_root,
            expected_config_sha256=expected,
            num_points=int(cfg.get('point_track_num_points', 1024)),
            horizon=int(cfg.get('point_track_horizon', 50)),
            cache_size=int(cfg.get('point_track_cache_size', 8)))
    row = dataset.curr_traj_data.iloc[current]
    if int(row['frame_index']) != current:
        raise ValueError('Point GT requires original unfiltered episode frame order')
    sample = dataset._point_track_store.read(
        metadata['source_prefix'], int(metadata['source_episode_index']), int(current),
        timestamp=float(row['timestamp']))
    return {'_point_track_targets': sample}
