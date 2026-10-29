"""Read existing evaluation artifacts only. Never imports a model or simulator."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def batch_starts(root):
    """Report the actual launcher assignments, without claiming UUID affinity."""
    rows=[]
    for path in sorted((root/'parallel-runs').glob('*/STARTED.json')):
        if not path.resolve().is_relative_to(root):continue
        data=read_json(path)
        rows.append({'file':str(path.relative_to(root)),
                     **{key:data.get(key) for key in
                        ('gpu','egl','tasks','command','hostname','started')}})
    return rows


def camera_stats(path):
    if not path.is_file():return None
    result={}
    with np.load(path,allow_pickle=False) as archive:
        for key in archive.files:
            if not key.startswith('video.'):continue
            image=archive[key]
            finite=bool(np.isfinite(image).all())
            result[key]={'shape':list(image.shape),'dtype':str(image.dtype),
                         'min':float(image.min()) if image.size and finite else None,
                         'max':float(image.max()) if image.size and finite else None,
                         'finite':finite,'nonzero':int(np.count_nonzero(image)),
                         'all_zero':bool(image.size and np.all(image==0))}
    return result


def inspect(output, task, episode):
    root=Path(output).resolve()
    if not root.is_dir():raise ValueError('Evaluation output directory does not exist')
    if task not in {row['task'] for row in json.loads((Path(__file__).resolve().parents[1]/'assets/eval_tasks.json').read_text())}:
        raise ValueError('Unknown benchmark task')
    if not 0<=episode<50:raise ValueError('Episode must be0..49')
    seeds=json.loads((Path(__file__).resolve().parents[1]/'assets/eval_scene_seeds_v1.json').read_text())
    expected_seed=seeds['scene_seeds'][task][episode]
    rows=[]
    for path in sorted(root.rglob('SIMULATION_ERROR.json')):
        relative=path.relative_to(root)
        if not path.resolve().is_relative_to(root):continue
        suffix=('episodes',task,str(episode))
        if tuple(path.parent.parts[-3:])!=suffix:continue
        error=read_json(path);directory=path.parent
        initial_path=directory/'env0.initial.json'
        if not initial_path.resolve().is_relative_to(root):raise ValueError('Artifact escapes output directory')
        initial=read_json(initial_path)
        def safe_stats(name):
            target=directory/name
            if not target.resolve().is_relative_to(root):raise ValueError('Artifact escapes output directory')
            return camera_stats(target)
        rows.append({'error_file':str(relative),'phase':error.get('phase'),
                     'reported_episode':error.get('episode'),'traceback':str(error.get('traceback',''))[-7000:],
                     'initial_observation_marker_exists':initial is not None,
                     'scene_seed':initial.get('scene_seed') if initial else None,
                     'scene_seed_matches':initial.get('scene_seed')==expected_seed if initial else None,
                     'initial_health':initial.get('health') if initial else None,
                     'failed_camera_arrays':safe_stats('failed_observation.npz'),
                     'initial_camera_arrays':safe_stats(f'episode{episode}.initial.npz')})
    identity_path=root/'EVALUATION_IDENTITY.json'
    identity=read_json(identity_path) if identity_path.resolve().is_relative_to(root) else None
    if identity is not None:
        identity={k:identity.get(k) for k in ('variant','checkpoint_sha256','source_manifest_sha256',
                    'scene_seed_definition_sha256','n_envs','tasks_per_gpu','rgb_schedule','initial_rgb','simulator_execution')}
    return {'format':'black_frame_artifact_inspection_v1','output':str(root),'task':task,'episode':episode,
            'expected_scene_seed':expected_seed,'run_identity':identity,'matching_errors':rows,
            'batch_launches':batch_starts(root),
            'status':'evidence_found' if rows else 'no_matching_public_pipeline_artifact',
            'limits':['A black array alone does not establish a rendering or GPU memory failure.',
                      'Missing initial arrays are normal for formal runs; inspect the initial health marker.',
                      'Recorded CUDA/EGL indices do not prove matching physical GPU UUIDs.',
                      'This command does not start a simulator, alter validation or prove the root cause.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--task',default='WaffleReheat');p.add_argument('--episode',type=int,default=0)
    a=p.parse_args();print(json.dumps(inspect(a.output,a.task,a.episode),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
