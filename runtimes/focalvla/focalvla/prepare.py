"""Full-data preparation: plan -> annotations -> teacher cache -> seal."""
import argparse
from collections import deque, Counter
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import numpy as np
from .common import read_json, atomic_json, digest, sha256, source_digest

CAMERAS = ('robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right')


def validate_labels(label):
    goals = label['goals']
    assert goals and goals[0]['start_frame'] == 0 and goals[-1]['end_frame'] == label['length']-1
    for i, goal in enumerate(goals):
        assert goal['text'].strip() and goal['entities'] and goal['start_frame'] <= goal['end_frame']
        assert all(0 <= e < len(label['entities']) for e in goal['entities'])
        if i: assert goals[i-1]['end_frame']+1 == goal['start_frame']
    if label.get('human_review') == 'rejected':
        raise ValueError('Annotation explicitly rejected; fix the label before caching.')


def annotate_one(row):
    from .annotation.prepare import prepare_case, D
    from .annotation.physics_labels_v4 import process
    from .annotation.finalize_labels import finish
    prepare_case(row)
    path = D/'physics_v5'/row['name']/'LABELS.json'
    if not path.exists():
        process(D/'cases'/row['name'])
        finish(row['name'])
    label = read_json(path)
    validate_labels(label)
    return {'episode':row['id'],'name':row['name'],'goals':len(label['goals']),
            'human_review':label.get('human_review','pending')}


class Videos:
    """One sequential decode per episode; bounded 51-frame lookahead."""
    def __init__(self, root, row):
        import cv2
        self.cv2 = cv2; self.caps = []; self.indices = [-1]*3; self.length = row['length']
        self.buffers = [deque(maxlen=51) for _ in CAMERAS]
        for cam in CAMERAS:
            key = 'videos/observation.images.'+cam
            file = Path(root)/row['dataset_name']/f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
            cap = cv2.VideoCapture(str(file))
            assert cap.isOpened(), file
            assert abs(cap.get(cv2.CAP_PROP_FPS)-20) < .01, ('Expected recorded 20 Hz video',file)
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000*row[key+'/from_timestamp'])
            self.caps.append(cap)

    def advance(self, camera, target):
        while self.indices[camera] < target:
            ok, frame = self.caps[camera].read()
            assert ok, (camera,target)
            self.indices[camera] += 1
            self.buffers[camera].append(self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB))

    def sample(self, t):
        self.advance(0,min(t+50,self.length-1))
        self.advance(1,t); self.advance(2,t)
        first = self.indices[0]-len(self.buffers[0])+1
        clip = np.stack([self.buffers[0][min(t+i,self.length-1)-first] for i in range(0,51,5)])
        return np.stack([clip[0],self.buffers[1][-1],self.buffers[2][-1]]), clip

    def close(self):
        for cap in self.caps: cap.release()


def cache_episode(row, frames, resources, root, teachers, identity):
    import mujoco
    import pyarrow.parquet as pq
    from .annotation.prepare import make_sim
    from .annotation.grounding import entity_geom_ids
    from .simulation import restore_state
    from .data_contract import repack
    label_path = root/'annotations/physics_v5'/row['name']/'LABELS.json'
    label = read_json(label_path); validate_labels(label)
    contract = digest({'inputs':identity,'label_sha256':sha256(label_path),'episode':row,'frames':frames.tolist()})
    dest = root/'cache'/f"{row['id']:06d}"; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        old = read_json(dest/'COMPLETE.json')
        if old['contract'] == contract:
            assert all((dest/f).stat().st_size == size for f,size in old['file_bytes'].items())
            return
        raise ValueError(f'Cache changed at {dest}; use a new prepared directory after annotation/config changes.')
    ep = Path(resources['source_root'])/row['source_prefix']/'lerobot/extras'/f"episode_{row['source_episode_index']:06d}"
    model, data, states, _ = make_sim(ep)
    assert len(states) == row['length'] == label['length']
    gids = entity_geom_ids(model,label['entities'])
    ds = Path(resources['dataset_root'])/row['dataset_name']
    table = pq.read_table(ds/f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet",
                          filters=[('episode_index','=',row['episode_index'])], columns=['observation.state','action','frame_index'])
    table = table.sort_by('frame_index')
    assert table['frame_index'].to_pylist() == list(range(row['length']))
    modality = read_json(ds/'meta/modality.json')
    robot = repack(np.array(table['observation.state'].to_pylist()),modality,'state')
    action = repack(np.array(table['action'].to_pylist()),modality,'action')
    assert robot.shape == (row['length'],16) and action.shape == (row['length'],12)
    np.save(dest/'state.npy',robot); np.save(dest/'action.npy',action); np.save(dest/'frames.npy',frames)
    count = len(frames)
    arrays = {'geo':np.zeros((count,2048),np.float32),'motion':np.zeros((count,256),np.float32),
              'geo_valid':np.zeros(count,bool),'motion_valid':np.zeros(count,bool),
              'goal_ids':np.zeros(count,np.int32)}
    videos = Videos(resources['dataset_root'],row)
    try:
        with mujoco.Renderer(model,height=256,width=256) as renderer:
            option = mujoco.MjvOption(); option.geomgroup[:]=0; option.geomgroup[1]=1
            for i,t in enumerate(frames):
                t = int(t)
                goal_id = next(j for j,g in enumerate(label['goals']) if g['start_frame'] <= t <= g['end_frame'])
                goal = label['goals'][goal_id]; arrays['goal_ids'][i] = goal_id
                relevant = sorted({gid for e in goal['entities'] for gid in gids[e]})
                assert relevant, (row['name'],goal)
                images, clip = videos.sample(t)
                restore_state(model,data,states[t]); mujoco.mj_forward(model,data)
                masks = []
                for cam in CAMERAS:
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    renderer.enable_segmentation_rendering(); seg = renderer.render().copy(); renderer.disable_segmentation_rendering()
                    masks.append(np.isin(seg[:,:,0],relevant)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
                target = teachers(images,clip,np.stack(masks))
                for key,value in target.items(): arrays[key][i] = value
                # RGB/teacher clips are transient. Training decodes original videos
                # in its bounded DataLoader pipeline; no full image copy is saved.
                if (i+1)%50 == 0: print({'episode':row['name'],'cached_frames':i+1,'total':count},flush=True)
    finally:
        videos.close()
    for key,value in arrays.items(): np.save(dest/(key+'.npy'),value)
    atomic_json(dest/'COMPLETE.json',{'contract':contract,'episode':row['id'],'frames':count,'goals':label['goals'],
                'label_sha256':sha256(label_path),'identity':identity,'image_storage':'source_video',
                'file_bytes':{name+'.npy':(dest/(name+'.npy')).stat().st_size for name in ('state','action','frames',*arrays)}})
    print({'event':'CACHED','episode':row['name'],'frames':count},flush=True)


def seal(config,resources,root):
    plan = read_json(root/'PLAN.json'); records = np.load(root/'records.npy',mmap_mode='r')
    repeats = np.load(root/'repeats.npy',mmap_mode='r')
    episodes = read_json(root/'episodes.json'); identity = read_json(root/'CACHE_IDENTITY.json')
    valid_all = np.zeros((len(records),2),bool)
    moments = {k:[np.zeros(d,np.float64),np.zeros(d,np.float64),0] for k,d in [('geo',2048),('motion',256)]}
    cache_manifest = []
    for row in episodes:
        eid = row['id']; lo,hi = np.searchsorted(records[:,0],[eid,eid+1]); dest = root/'cache'/f'{eid:06d}'
        if lo == hi: continue
        meta = read_json(dest/'COMPLETE.json'); assert meta['frames'] == hi-lo and meta['identity'] == identity
        label_path = root/'annotations/physics_v5'/row['name']/'LABELS.json'
        validate_labels(read_json(label_path)); assert sha256(label_path) == meta['label_sha256']
        assert all((dest/f).stat().st_size == size for f,size in meta['file_bytes'].items())
        assert np.array_equal(np.load(dest/'frames.npy'),records[lo:hi,1])
        cache_manifest.append({'episode':eid,'complete_sha256':sha256(dest/'COMPLETE.json')})
        for col,key in enumerate(('geo','motion')):
            x = np.load(dest/(key+'.npy')).astype(np.float64)
            valid = np.load(dest/(key+'_valid.npy')); assert np.isfinite(x).all()
            valid_all[lo:hi,col] = valid
            w = repeats[lo:hi]*valid
            moments[key][0] += (x*w[:,None]).sum(0)
            moments[key][1] += (x*x*w[:,None]).sum(0)
            moments[key][2] += int(w.sum())
    stats = {}
    for key,(total,square,n) in moments.items():
        assert n > 0, f'No valid {key} supervision'
        mean = total/n; std = np.sqrt(np.maximum(0,square/n-mean*mean))
        stats[key] = {'mean':mean.tolist(),'std':np.maximum(std,1e-6).tolist(),'valid_training_draws':n}
    np.save(root/'validity.npy',valid_all); atomic_json(root/'target_stats.json',stats)
    atomic_json(root/'CACHE_MANIFEST.json',cache_manifest)
    atomic_json(root/'READY.json',{'plan_contract':plan['contract'],'total_steps':config['total_steps'],'global_batch':config['global_batch'],
        'episodes':len(episodes),'tasks':len({r['task_name'] for r in episodes}),'missing_episodes':0,'invalid_labels':0,
        'valid_geo_draws':moments['geo'][2],'valid_motion_draws':moments['motion'][2],
        'norm_sha256':sha256(Path(resources['norm_assets'])/'norm_stats.json'),'tokenizer_sha256':sha256(resources['tokenizer']),
        'training_files':{f:sha256(root/f) for f in ('draws.npy','records.npy','episodes.json','target_stats.json','validity.npy','CACHE_MANIFEST.json')},
        'annotation_review':'Automated integrity checks; human_review status is retained per episode, not fabricated.',
        'teachers_frozen':True,'future_visible_to_policy':False,'training_images':'source_video'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','annotate','cache','seal'))
    p.add_argument('--config',required=True); p.add_argument('--resources',required=True); p.add_argument('--prepared',required=True)
    p.add_argument('--workers',type=int,default=8)
    args = p.parse_args(); config = read_json(args.config); resources = read_json(args.resources); root = Path(args.prepared)
    os.environ['FOCAL_RESOURCES'] = str(Path(args.resources).resolve()); os.environ['FOCAL_PREPARED'] = str(root.resolve())
    os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1'); os.environ.setdefault('LP_NUM_THREADS','2')
    if args.stage == 'plan':
        from .plan import build
        print(build(config,resources,root)); return
    episodes = read_json(root/'episodes.json')
    if args.stage == 'annotate':
        results = []; failures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = {pool.submit(annotate_one,row):row for row in episodes}
            for future in concurrent.futures.as_completed(jobs):
                row = jobs[future]
                try: results.append(future.result())
                except Exception as exc: failures.append({'episode':row['name'],'error':repr(exc)})
        atomic_json(root/'ANNOTATION_QC.json',{'episodes':len(episodes),'completed':len(results),'failures':failures,
                     'human_review':dict(Counter(r['human_review'] for r in results))})
        if failures: raise RuntimeError(f"{len(failures)} annotations require fixes; see ANNOTATION_QC.json")
    elif args.stage == 'cache':
        import torch
        rank,world,local = (int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')])
        torch.cuda.set_device(local); torch.set_num_threads(4)
        from .teachers import Teachers
        # Weight identities are hashed once per worker before expensive extraction.
        identity = {'source_sha256':source_digest(),'plan_contract':read_json(root/'PLAN.json')['contract'],
                    'weights':{k:sha256(resources[k]) for k in ('vggt_weights','track4world_weights')},
                    'da3_sha256':sha256(Path(resources['da3_weights'])/'model.safetensors')}
        if rank == 0:
            if (root/'CACHE_IDENTITY.json').exists(): assert read_json(root/'CACHE_IDENTITY.json') == identity
            atomic_json(root/'CACHE_IDENTITY.json',identity)
        teachers = Teachers(resources)
        records = np.load(root/'records.npy',mmap_mode='r')
        for row in episodes[rank::world]:
            lo,hi = np.searchsorted(records[:,0],[row['id'],row['id']+1])
            if lo < hi: cache_episode(row,records[lo:hi,1],resources,root,teachers,identity)
    else:
        seal(config,resources,root)


if __name__ == '__main__':
    main()
