"""Full-scope cache preparation using the same FOCAL targets, without RGB files."""
import argparse,concurrent.futures,fcntl,multiprocessing,os,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'runtimes/focalvla'))
from focalvla.common import read_json,atomic_json,digest,sha256,source_digest
from prepare_stream import PreparedStream


def annotate(row):
    from focalvla.prepare import annotate_one
    return annotate_one(row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources',required=True);parser.add_argument('--prepared',required=True)
    parser.add_argument('--status-dir',required=True);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--renderer',choices=['osmesa','egl'],default='osmesa')
    parser.add_argument('--annotation-workers',type=int,default=4)
    parser.add_argument('--max-episodes',type=int,default=0,help='Bounded integration check only; 0 uses the complete scope.')
    parser.add_argument('--episode-limit',type=int,default=0,help='Restrict assigned episodes for integration validation; 0 uses all.')
    args=parser.parse_args();root=Path(args.prepared);dest=Path(args.status_dir);dest.mkdir(parents=True,exist_ok=True)
    if args.batch_size<1 or args.annotation_workers<1 or min(args.max_episodes,args.episode_limit)<0:
        parser.error('Batch size/workers must be positive; episode limits must be nonnegative.')
    os.environ.update(FOCAL_RESOURCES=str(Path(args.resources).resolve()),FOCAL_PREPARED=str(root.resolve()),
                      MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    resources=read_json(args.resources)
    expected={'source_sha256':source_digest(),'plan_contract':read_json(root/'PLAN.json')['contract']}
    with (root/'cache_identity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'CACHE_IDENTITY.json').exists():
            identity=read_json(root/'CACHE_IDENTITY.json')
            assert all(identity[k]==v for k,v in expected.items())
        else:
            identity={**expected,'weights':{k:sha256(resources[k]) for k in ('vggt_weights','track4world_weights')},
                      'da3_sha256':sha256(Path(resources['da3_weights'])/'model.safetensors')}
            atomic_json(root/'CACHE_IDENTITY.json',identity)
    records=np.load(root/'records.npy',mmap_mode='r');episodes=read_json(root/'episodes.json')[rank::world]
    if args.episode_limit:episodes=episodes[:args.episode_limit]
    pending=[];completed=0;targets_done=0
    for row in episodes:
        marker=root/'cache'/f"{row['id']:06d}"/'COMPLETE.json'
        if marker.exists():
            info=read_json(marker)
            assert info['identity']==identity and info.get('image_storage')=='source_video'
            if not all((marker.parent/f).exists() and (marker.parent/f).stat().st_size==n for f,n in info['file_bytes'].items()):
                pending.append(row);continue
            lo,hi=np.searchsorted(records[:,0],[row['id'],row['id']+1])
            label=root/'annotations/physics_v5'/row['name']/'LABELS.json'
            assert info['contract']==digest({'inputs':identity,'label_sha256':sha256(label),'episode':row,'frames':records[lo:hi,1].tolist()})
            completed+=1;targets_done+=info['frames']
        else:pending.append(row)
    if args.max_episodes:pending=pending[:args.max_episodes]
    execution={'version':'feature_only_stream_v1','batch_size':args.batch_size,'renderer':args.renderer,
               'sources':{name:sha256(Path(__file__).parent/name) for name in ['feature_teachers.py','prepare_stream.py','prepare_fast.py']}}
    def status(stage,**fields):
        atomic_json(dest/f'prepare_rank{rank}.json',{'rank':rank,'world':world,'time':time.time(),'hostname':os.uname().nodename,
            'stage':stage,'completed_episodes':completed,'assigned_episodes':len(episodes),'completed_targets':targets_done,'execution':execution,**fields})
    status('loading_teachers',remaining_episodes=len(pending))
    if not pending:status('complete');return
    import torch
    from feature_teachers import FeatureTeachers
    torch.cuda.set_device(local);torch.set_num_threads(4)
    stream=None
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.annotation_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={}
            def fill(start):
                for i in range(start,min(start+8,len(pending))):
                    if i not in futures:futures[i]=pool.submit(annotate,pending[i])
            def submit(i):
                row=pending[i];futures.pop(i).result();fill(i+1)
                lo,hi=np.searchsorted(records[:,0],[row['id'],row['id']+1])
                stream.submit(row,np.array(records[lo:hi,1]))
            fill(0)
            stream=PreparedStream(resources,root,args.batch_size,args.renderer,local)
            submit(0)
            teachers=FeatureTeachers(resources,static_grid=True,prune_outputs=True,cast_linears=True)
            for position,row in enumerate(pending):
                if position+1<len(pending):submit(position+1)
                start=time.perf_counter();wait=0.;gpu=0.;count=0
                kind,meta=stream.get();assert kind=='begin' and meta['row']['id']==row['id']
                cache=root/'cache'/f"{row['id']:06d}";cache.mkdir(parents=True,exist_ok=True)
                (cache/'COMPLETE.json').unlink(missing_ok=True)
                frames=meta['frames'];n=len(frames)
                for key in ['state','action','frames']:np.save(cache/(key+'.npy'),meta[key])
                values={'geo':np.zeros((n,2048),np.float32),'motion':np.zeros((n,256),np.float32),
                        'geo_valid':np.zeros(n,bool),'motion_valid':np.zeros(n,bool),'goal_ids':np.zeros(n,np.int32)}
                status('teacher_cache',episode=row['name'],frames=n,cached_frames=0)
                while True:
                    began=time.perf_counter();kind,data=stream.get();wait+=time.perf_counter()-began
                    if kind=='end':assert data==row['id'];break
                    assert kind=='batch' and data['episode']==row['id']
                    begin=data['start'];size=len(data['samples']);assert begin==count
                    began=time.perf_counter();target=teachers(data['samples']);gpu+=time.perf_counter()-began
                    for key,array in target.items():values[key][begin:begin+size]=array
                    values['goal_ids'][begin:begin+size]=data['goals'];count+=size
                    status('teacher_cache',episode=row['name'],frames=n,cached_frames=count,
                           seconds_per_target=(time.perf_counter()-start)/count,teacher_seconds_per_target=gpu/count,
                           input_wait_seconds_per_target=wait/count,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
                assert count==n
                for key,array in values.items():np.save(cache/(key+'.npy'),array)
                contract=digest({'inputs':identity,'label_sha256':meta['label_sha256'],'episode':row,'frames':frames.tolist()})
                atomic_json(cache/'COMPLETE.json',{'contract':contract,'episode':row['id'],'frames':n,'goals':meta['label']['goals'],
                    'label_sha256':meta['label_sha256'],'identity':identity,'image_storage':'source_video','execution':execution,
                    'file_bytes':{name+'.npy':(cache/(name+'.npy')).stat().st_size for name in ['state','action','frames',*values]}})
                completed+=1;targets_done+=n
                event={'event':'CACHED_FAST','rank':rank,'episode':row['name'],'frames':n,'seconds':time.perf_counter()-start,
                       'teacher_seconds':gpu,'input_wait_seconds':wait,'time':time.time()}
                print(event,flush=True)
                with (dest/f'cache_performance_rank{rank}.jsonl').open('a') as output:
                    import json
                    output.write(json.dumps(event)+'\n')
                status('episode_complete',episode=row['name'])
        status('complete')
    finally:
        if stream is not None:stream.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        # Torchrun owns peer shutdown; native SCO recovery restarts completed work.
        traceback.print_exc();raise
