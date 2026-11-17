"""Reverse-order feature extraction into an isolated numeric cache."""
import argparse,concurrent.futures,multiprocessing,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'runtimes/focalvla'))
from focalvla.common import read_json,atomic_json,digest,sha256
from cache_partition import assigned_episodes
from prepare_stream import PreparedStream
from assist_cache import complete,label_path,atomic,coverage

def annotate(row,primary,helper):
    from focalvla.prepare import annotate_one,validate_labels
    dest=label_path(helper,row);source=label_path(primary,row)
    if not dest.exists() and source.exists():
        data=source.read_bytes()
        import json
        validate_labels(json.loads(data));atomic(dest,data)
    return annotate_one(row)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resources',required=True);p.add_argument('--primary',required=True)
    p.add_argument('--prepared',required=True);p.add_argument('--status-dir',required=True);p.add_argument('--shard-index',type=int,required=True)
    a=p.parse_args();primary=Path(a.primary);root=Path(a.prepared);dest=Path(a.status_dir)
    assert primary.resolve()!=root.resolve() and 0<=a.shard_index<5
    os.environ.update(FOCAL_RESOURCES=str(Path(a.resources).resolve()),FOCAL_PREPARED=str(root.resolve()),
        MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2')
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==2
    resources=read_json(a.resources);identity=read_json(primary/'CACHE_IDENTITY.json')
    records=np.load(root/'records.npy',mmap_mode='r')
    rows=assigned_episodes(read_json(root/'episodes.json'),rank,2,a.shard_index,5)
    pending=list(reversed(rows));skipped=0;written=0;stream=None
    execution={'version':'isolated_reverse_assist_v1','shard_index':a.shard_index,'rank':rank,'batch_size':8,
        'sources':{n:sha256(Path(__file__).parent/n) for n in ['prepare_assist.py','assist_cache.py','feature_teachers.py','prepare_stream.py']}}
    def status(stage,**kw):
        atomic_json(dest/f'assist_rank{rank}.json',{'time':time.time(),'hostname':os.uname().nodename,'rank':rank,
          'shard_index':a.shard_index,'stage':stage,'skipped_complete':skipped,'written_episodes':written,'execution':execution,**kw})
    def frames(row):
        lo,hi=np.searchsorted(records[:,0],[row['id'],row['id']+1]);return np.array(records[lo:hi,1])
    status('scan')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        cursor=iter(pending)
        def candidate():
            nonlocal skipped
            for row in cursor:
                f=frames(row)
                if complete(primary,row,f,identity) or complete(root,row,f,identity):skipped+=1;continue
                return row,f,pool.submit(annotate,row,primary,root)
            return None
        current=candidate();following=candidate()
        if current is not None:
            import torch
            from feature_teachers import FeatureTeachers
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(4)
            stream=PreparedStream(resources,root,8,'osmesa',int(os.environ['LOCAL_RANK']))
            current[2].result();stream.submit(current[0],current[1]);status('loading_teachers')
            teachers=FeatureTeachers(resources,static_grid=True,prune_outputs=True,cast_linears=True)
            try:
                while current is not None:
                    if following is not None:following[2].result();stream.submit(following[0],following[1])
                    after=candidate();row,f,_=current
                    kind,meta=stream.get();assert kind=='begin' and meta['row']['id']==row['id']
                    skip=bool(complete(primary,row,f,identity));cache=root/'cache'/f"{row['id']:06d}"
                    n=len(f);count=0;began=time.perf_counter()
                    if not skip:
                        cache.mkdir(parents=True,exist_ok=True);(cache/'COMPLETE.json').unlink(missing_ok=True)
                        for key in ['state','action','frames']:np.save(cache/(key+'.npy'),meta[key])
                        values={'geo':np.zeros((n,2048),np.float32),'motion':np.zeros((n,256),np.float32),
                          'geo_valid':np.zeros(n,bool),'motion_valid':np.zeros(n,bool),'goal_ids':np.zeros(n,np.int32)}
                    while True:
                        kind,data=stream.get()
                        if kind=='end':assert data==row['id'];break
                        assert kind=='batch' and data['episode']==row['id']
                        if skip:continue
                        begin=data['start'];size=len(data['samples']);assert begin==count
                        target=teachers(data['samples'])
                        for key,value in target.items():values[key][begin:begin+size]=value
                        values['goal_ids'][begin:begin+size]=data['goals'];count+=size
                        status('teacher_cache',episode=row['name'],cached_frames=count,frames=n,seconds_per_target=(time.perf_counter()-began)/count)
                    if skip:skipped+=1
                    else:
                        assert count==n
                        for key,value in values.items():np.save(cache/(key+'.npy'),value)
                        contract=digest({'inputs':identity,'label_sha256':meta['label_sha256'],'episode':row,'frames':f.tolist()})
                        atomic_json(cache/'COMPLETE.json',{'contract':contract,'episode':row['id'],'frames':n,'goals':meta['label']['goals'],
                          'label_sha256':meta['label_sha256'],'identity':identity,'image_storage':'source_video','execution':execution,
                          'file_bytes':{key+'.npy':(cache/(key+'.npy')).stat().st_size for key in ['state','action','frames',*values]}})
                        written+=1
                        print('ASSIST_CACHED',{'shard':a.shard_index,'rank':rank,'episode':row['name'],'frames':n,'seconds':time.perf_counter()-began},flush=True)
                    current,following=following,after
            finally:stream.close()
    proof=coverage(primary,root,rows,records,identity);assert proof is not None
    atomic_json(dest/f'assist_rank{rank}_DONE.json',proof);status('complete',covered_episodes=proof['episodes'])

if __name__=='__main__':main()
