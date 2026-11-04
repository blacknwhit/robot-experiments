"""Counter-indexed full dataset: prefetch does not advance the committed cursor."""
import random
import numpy as np
from torch.utils.data import Dataset,Sampler,DataLoader

class WindowDataset(Dataset):
 def __init__(self,c):
  from omegaconf import OmegaConf
  from starVLA.dataloader.lerobot_datasets import get_vla_dataset
  cfg=OmegaConf.create(c['framework_config']['datasets']['vla_data'])
  self.base=get_vla_dataset(cfg,seed=c['seed']);self.base.set_epoch(0);self.seed=c['seed']
 def __len__(self):return 70000*64
 def __getitem__(self,index):
  # Dataset-side transforms have the same seed regardless of worker/prefetch.
  import torch
  seed=(self.seed+int(index)*1009)%(2**32)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  sample=self.base[int(index)]
  assert sample['action'].shape==(50,12)
  assert sample['point_delta'].shape==(1024,50,3)
  assert sample['point_meta']['coordinate_frame']=='window_start_robot_base'
  assert sample['point_meta']['units']=='meters'
  # Source frame0->frame1 may span two physical control intervals. Keep the
  # baseline action draw; exclude this whole auxiliary window, not just delta0.
  if int(sample['point_meta']['frame_index'])==0:
   sample['point_valid']=np.zeros_like(sample['point_valid'],dtype=bool)
  sample['_c3_global_sample_index']=int(index)
  return sample

class RankIndices(Sampler):
 def __init__(self,start,stop,rank,world,batch=64):
  assert batch%world==0 and 0<=rank<world and 0<=start<=stop
  self.start,self.stop,self.rank,self.world,self.batch=start,stop,rank,world,batch
 def __iter__(self):
  n=self.batch//self.world
  for step in range(self.start,self.stop):yield from range(step*self.batch+self.rank*n,step*self.batch+(self.rank+1)*n)
 def __len__(self):return (self.stop-self.start)*self.batch//self.world

def collate(rows):return rows
def loader(c,start,stop,rank,world):
 import torch
 ds=WindowDataset(c);workers=c['data_workers']
 options=dict(batch_size=c['micro_batch'],sampler=RankIndices(start,stop,rank,world),
              num_workers=workers,collate_fn=collate,pin_memory=False,drop_last=True,
              generator=torch.Generator().manual_seed(c['seed']+rank))
 if workers:options.update(prefetch_factor=c['prefetch_factor'],persistent_workers=True,multiprocessing_context='spawn')
 return DataLoader(ds,**options)
