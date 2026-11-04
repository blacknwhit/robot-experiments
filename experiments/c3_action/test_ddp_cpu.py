"""Two CPU/Gloo ranks: real DDP full-state checkpoint and restart parity."""
from pathlib import Path
import os,tempfile
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import state

class Toy(nn.Module):
 def __init__(self):super().__init__();self.linear=nn.Linear(3,2);self.unused=nn.Parameter(torch.ones(2))
 def forward(self,x):return self.linear(x)

def worker(rank,path):
 root=Path(path);dist.init_process_group('gloo',init_method=(root/'rendezvous').resolve().as_uri(),rank=rank,world_size=2)
 torch.manual_seed(42);m=Toy();d=DDP(m,find_unused_parameters=True,gradient_as_bucket_view=True)
 opt=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False)
 torch.manual_seed(100+rank)
 def step(ddp,optimizer):
  optimizer.zero_grad();loss=ddp(torch.randn(4,3)).square().mean();loss.backward();optimizer.step()
 step(d,opt);step(d,opt);cp=state.save(root/'checkpoints',m,opt,2,'cpu-ddp',rank,2)
 step(d,opt);want=state.tree_signature({'model':m.state_dict(),'optimizer':opt.state_dict()})
 n=Toy();nd=DDP(n,find_unused_parameters=True,gradient_as_bucket_view=True);no=torch.optim.AdamW(n.parameters(),lr=.001,foreach=False)
 assert state.load(cp,n,no,'cpu-ddp',2,rank)==2
 step(nd,no);assert state.tree_signature({'model':n.state_dict(),'optimizer':no.state_dict()})==want
 assert n.unused not in no.state
 (root/f'PASS_RANK{rank}').write_text(want);dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 with tempfile.TemporaryDirectory() as path:
  mp.spawn(worker,args=(path,),nprocs=2,join=True)
  assert (Path(path)/'PASS_RANK0').read_text()==(Path(path)/'PASS_RANK1').read_text()
 print('TWO_RANK_CPU_DDP_RESUME_EXACT')
