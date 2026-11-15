"""Persistent bounded producer of exact original-video clips and entity masks."""
from collections import deque
from pathlib import Path
import multiprocessing as mp
import os,queue,time,traceback
import numpy as np
from focalvla.common import read_json,sha256


class EpisodeReader:
    def __init__(self,row,frames,resources,root):
        import cv2,mujoco,pyarrow.parquet as pq
        from focalvla.annotation.prepare import make_sim
        from focalvla.annotation.grounding import entity_geom_ids
        from focalvla.prepare import Videos,validate_labels
        from focalvla.data_contract import repack
        cv2.setNumThreads(1)
        self.row=row;self.frames=frames;self.mujoco=mujoco
        label_path=Path(root)/'annotations/physics_v5'/row['name']/'LABELS.json'
        self.label=read_json(label_path);validate_labels(self.label)
        ep=Path(resources['source_root'])/row['source_prefix']/'lerobot/extras'/f"episode_{row['source_episode_index']:06d}"
        self.model,self.data,self.states,_=make_sim(ep)
        assert len(self.states)==row['length']==self.label['length']
        gids=entity_geom_ids(self.model,self.label['entities'])
        self.relevant=[np.array(sorted({g for e in goal['entities'] for g in gids[e]}),dtype=np.int32) for goal in self.label['goals']]
        assert all(len(g) for g in self.relevant)
        ds=Path(resources['dataset_root'])/row['dataset_name']
        table=pq.read_table(ds/f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet",
            filters=[('episode_index','=',row['episode_index'])],columns=['observation.state','action','frame_index']).sort_by('frame_index')
        assert table['frame_index'].to_pylist()==list(range(row['length']))
        modality=read_json(ds/'meta/modality.json')
        robot=repack(np.array(table['observation.state'].to_pylist()),modality,'state')
        action=repack(np.array(table['action'].to_pylist()),modality,'action')
        assert robot.shape==(row['length'],16) and action.shape==(row['length'],12)
        self.metadata={'row':row,'frames':frames,'state':robot,'action':action,'label':self.label,'label_sha256':sha256(label_path)}
        self.videos=Videos(resources['dataset_root'],row)
        self.videos.buffers[1]=deque(maxlen=1);self.videos.buffers[2]=deque(maxlen=1)
        self.renderer=mujoco.Renderer(self.model,height=256,width=256)
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[1]=1

    def sample(self,index):
        from focalvla.prepare import CAMERAS
        from focalvla.simulation import restore_state
        t=int(self.frames[index]);mujoco=self.mujoco
        goal_id=next(i for i,g in enumerate(self.label['goals']) if g['start_frame']<=t<=g['end_frame'])
        begin=time.perf_counter();images,clip=self.videos.sample(t);video=time.perf_counter()-begin
        begin=time.perf_counter();restore_state(self.model,self.data,self.states[t]);mujoco.mj_forward(self.model,self.data)
        masks=[]
        for camera in CAMERAS:
            self.renderer.update_scene(self.data,camera=camera,scene_option=self.option)
            self.renderer.enable_segmentation_rendering();seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
            masks.append(np.isin(seg[:,:,0],self.relevant[goal_id])&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
        return (images,clip,np.stack(masks)),goal_id,{'video':video,'mask':time.perf_counter()-begin}

    def close(self):
        self.renderer.close();self.videos.close()


def _producer(commands,output,stop,resources,root,batch_size,renderer,device):
    os.environ.update(MUJOCO_GL=renderer,PYOPENGL_PLATFORM=renderer,MUJOCO_EGL_DEVICE_ID=str(device),
                      LIBGL_ALWAYS_SOFTWARE='1' if renderer=='osmesa' else '0',LP_NUM_THREADS='2')
    def send(item):
        while not stop.is_set():
            try:output.put(item,timeout=.5);return True
            except queue.Full:pass
        return False
    try:
        while not stop.is_set():
            try:command=commands.get(timeout=.5)
            except queue.Empty:continue
            if command is None:return
            row,frames=command
            reader=EpisodeReader(row,frames,resources,root)
            try:
                if not send(('begin',reader.metadata)):return
                for start in range(0,len(frames),batch_size):
                    if stop.is_set():return
                    samples=[];goals=[];timing={'video':0.,'mask':0.}
                    for i in range(start,min(start+batch_size,len(frames))):
                        sample,goal,times=reader.sample(i);samples.append(sample);goals.append(goal)
                        for key in timing:timing[key]+=times[key]
                    if not send(('batch',{'episode':row['id'],'start':start,'samples':samples,'goals':goals,'timing':timing})):return
                if not send(('end',row['id'])):return
            finally:reader.close()
    except BaseException:
        send(('error',traceback.format_exc()))
        raise


class PreparedStream:
    def __init__(self,resources,root,batch_size=8,renderer='osmesa',device=0):
        context=mp.get_context('spawn')
        self.commands=context.Queue(maxsize=1);self.output=context.Queue(maxsize=2);self.stop=context.Event()
        self.process=context.Process(target=_producer,args=(self.commands,self.output,self.stop,resources,root,batch_size,renderer,device),daemon=True)
        self.process.start()

    def submit(self,row,frames):
        while True:
            if not self.process.is_alive():raise RuntimeError('Input producer exited before accepting an episode')
            try:self.commands.put((row,frames),timeout=.5);return
            except queue.Full:pass

    def get(self):
        while True:
            try:
                kind,value=self.output.get(timeout=1.)
                if kind=='error':raise RuntimeError(value)
                return kind,value
            except queue.Empty:
                if not self.process.is_alive():raise RuntimeError(f'Input producer exited ({self.process.exitcode})')

    def close(self):
        self.stop.set()
        # Drain IPC while the worker exits: Queue feeder threads may otherwise
        # hold the process alive after the consumer has intentionally stopped.
        deadline=time.monotonic()+15
        while self.process.is_alive() and time.monotonic()<deadline:
            try:self.output.get(timeout=.2)
            except queue.Empty:pass
            self.process.join(timeout=.1)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        for q in (self.commands,self.output):q.cancel_join_thread();q.close()
