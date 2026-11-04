from pathlib import Path
import os,json,random,time,hashlib
import numpy as np
import torch
import torch.distributed as dist
from contract import write,sha256 as digest_file
from checkpoints import compatible_checkpoint
def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return x
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
def restore_rng(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
 if x['cuda'] is not None:torch.cuda.set_rng_state(x['cuda'])
def barrier():
 if dist.is_initialized():dist.barrier()
def tree_signature(x):
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,torch.Tensor):
   t=v.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(tuple,list)):
   for item in v:visit(item)
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()
def save(root,model,optimizer,step,fingerprint,rank,world):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/f'step_{step:06d}';tmp=root/f'step_{step:06d}.tmp'
 if rank==0:
  if path.exists():raise RuntimeError('Refuse to overwrite checkpoint '+str(path))
  # A prior killed writer may leave an incomplete temporary; never reuse it.
  if tmp.exists():tmp.rename(root/f'step_{step:06d}.incomplete.{int(time.time())}')
  tmp.mkdir()
 barrier();torch.save(rng_state(),tmp/f'rng_rank{rank}.pt')
 signature={'model':tree_signature(model.state_dict()),'optimizer':tree_signature(optimizer.state_dict())}
 signatures=[None]*world
 if world>1:dist.all_gather_object(signatures,signature)
 else:signatures=[signature]
 assert all(s==signatures[0] for s in signatures),'DDP model/Adam replicas diverged'
 barrier()
 start=time.monotonic()
 if rank==0:
  torch.save({'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'step':step,'next_global_update':step,'fingerprint':fingerprint,'world_size':world},tmp/'training.pt')
  files={p.name:{'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in tmp.iterdir() if p.is_file()}
  write(tmp/'COMPLETE.json',{'complete':True,'step':step,'fingerprint':fingerprint,'world_size':world,'rank_rng':list(range(world)),'ddp_replica_signatures':signatures,'files':files,'save_seconds':time.monotonic()-start})
  os.replace(tmp,path)
 barrier();return path
def load(path,model,optimizer,fingerprint,world,rank):
 proof=compatible_checkpoint(path,fingerprint,world,verify_hashes=True);assert proof
 x=torch.load(Path(path)/'training.pt',map_location='cpu',weights_only=False,mmap=True)
 assert x['fingerprint']==fingerprint and x['world_size']==world and x['step']==x['next_global_update']==proof['step']
 model.load_state_dict(x['model'],strict=True);optimizer.load_state_dict(x['optimizer'])
 # Value checks distinguish successful deserialization from actual model/Adam restoration.
 for name,v in model.state_dict().items():
  assert torch.equal(v.detach().cpu(),x['model'][name]),name
 saved=x['optimizer'];actual=optimizer.state_dict();assert actual['param_groups']==saved['param_groups']
 for pid,values in actual['state'].items():
  for name,v in values.items():
   ref=saved['state'][pid][name]
   assert torch.equal(v.detach().cpu(),ref) if isinstance(v,torch.Tensor) else v==ref
 rng=torch.load(Path(path)/f'rng_rank{rank}.pt',weights_only=False);restore_rng(rng)
 assert torch.equal(torch.get_rng_state(),rng['torch']) and (rng['cuda'] is None or torch.equal(torch.cuda.get_rng_state(),rng['cuda']))
 return x['step']
