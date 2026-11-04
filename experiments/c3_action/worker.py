"""C3 FP32-parameter/native-AdamW DDP training with BF16 autocast, no EMA."""
import pyarrow  # Keep accepted import order before Torch.
import argparse,copy,json,math,os,socket,time
from pathlib import Path
from datetime import timedelta
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contract import activate,digest,fingerprint,sha256,write
activate()
from model import JointPolicy
from checkpoints import latest_checkpoint,compatible_checkpoint
import state

def lr_factor(step,c):
 if step<c['warmup_steps']:return (step+1)/(c['warmup_steps']+1)
 ratio=(step-c['warmup_steps'])/(c['schedule_steps']-c['warmup_steps']);floor=c['min_learning_rate']/c['learning_rate']
 return floor+(1-floor)*.5*(1+math.cos(math.pi*min(ratio,1.)))

def export_policy(model,run,step,fp):
 from safetensors.torch import save_file
 root=run/'exports'/f'step_{step:06d}';root.mkdir(parents=True,exist_ok=True)
 path=root/'model.safetensors';marker=root/'COMPLETE.json'
 if marker.exists():
  m=json.loads(marker.read_text());assert m['fingerprint']==fp and m['step']==step and sha256(path)==m['sha256'];return
 tmp=root/f'model.{os.getpid()}.partial';save_file(model.export_policy_state(),str(tmp));h=sha256(tmp);os.replace(tmp,path)
 write(marker,{'step':step,'fingerprint':fp,'policy_only':True,'sha256':h})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--stop',type=int,required=True);a=ap.parse_args()
 c=json.loads(a.config.read_text());run=Path(c['run']);fp=fingerprint(c)
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
 assert world==c['world'];torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
 assert 'H100' in torch.cuda.get_device_name(local),'This release is provisioned for H10080GB, not memory-sharded 5090 training'
 assert c['mode']=='smoke' or (world==8 and c['accumulation']==1 and a.stop==70000)
 if (run/'FINISHED.json').exists():
  done=json.loads((run/'FINISHED.json').read_text());assert done['fingerprint']==fp
  if done['step']>=a.stop:
   if rank==0:
    assert compatible_checkpoint(run/'checkpoints'/f"step_{done['step']:06d}",fp,world)
    ex=run/'exports'/f"step_{done['step']:06d}";m=json.loads((ex/'COMPLETE.json').read_text())
    assert m['fingerprint']==fp and sha256(ex/'model.safetensors')==m['sha256']
   dist.barrier();dist.destroy_process_group();return
 if rank==0:
  assert sha256(c['framework_config']['framework']['pi05']['init_checkpoint'])==c['base_sha256']
  selection=latest_checkpoint(run/'checkpoints',fp,world)
  path,m,rejected=selection
  if path is None and (run/'CLAIM.json').exists():raise RuntimeError('No complete compatible state after prior start; never reset base')
  if path is None:write(run/'CLAIM.json',{'fingerprint':fp,'started_unix':time.time(),'world':world})
  write(run/'recipe.json',c)
  chosen=[str(path) if path else None,rejected]
 else:chosen=[None,None]
 dist.broadcast_object_list(chosen,src=0);last=Path(chosen[0]) if chosen[0] else None
 model=JointPolicy(c['arm'],c).cuda();main,aux=model.parameters_by_group()
 groups=[{'params':main,'lr':c['learning_rate'],'name':'main'}]
 if aux:groups.append({'params':aux,'lr':c['head_learning_rate'],'name':'aux'})
 optimizer=torch.optim.AdamW(groups,betas=tuple(c['optimizer']['betas']),eps=c['optimizer']['eps'],
     weight_decay=c['optimizer']['weight_decay'],foreach=False)
 start=state.load(last,model,optimizer,fp,world,rank) if last else 0
 attempt={'host':socket.gethostname(),'pid':os.getpid(),'started_unix':time.time(),'start_step':start,
          'world':world,'rank':rank,'fingerprint':fp,'gpu':torch.cuda.get_device_name(local),
          'model_optimizer_rng_exact':bool(last),'next_global_update':start,'checkpoint':str(last) if last else None}
 write(run/f'ATTEMPT_RANK{rank}.json',attempt)
 wrapped=DDP(model,device_ids=[local],output_device=local,gradient_as_bucket_view=True,
             find_unused_parameters=True,broadcast_buffers=False,bucket_cap_mb=25)
 from data import loader
 batches=iter(loader(c,start,a.stop,rank,world)) if start<a.stop else None
 model.train();history=run/f'metrics_rank{rank}.jsonl'
 for step in range(start,a.stop):
  factor=lr_factor(step,c)
  for g in optimizer.param_groups:g['lr']=(c['learning_rate'] if g['name']=='main' else c['head_learning_rate'])*factor
  weight=c['aux_lambda']*min(1.,(step+1)/c['aux_warmup_steps']) if c['arm']!='C0' else 0.
  optimizer.zero_grad(set_to_none=True);begin=time.monotonic();action=auxloss=0.;indices=[]
  for mi in range(c['accumulation']):
   rows=next(batches);indices.extend(int(x['_c3_global_sample_index']) for x in rows)
   sync=wrapped.no_sync() if mi<c['accumulation']-1 else nullcontext()
   with sync:
    losses=wrapped(rows,noise_seed=c['seed']+step*10000+rank*100+mi,augment=True)
    loss=(losses['action']+weight*losses['aux'])/c['accumulation']
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward()
   action+=float(losses['action'].detach())/c['accumulation'];auxloss+=float(losses['aux'].detach())/c['accumulation'];model.close_capture()
  main_norm=torch.nn.utils.clip_grad_norm_(main,c['clip_main'],error_if_nonfinite=True)
  aux_norm=torch.nn.utils.clip_grad_norm_(aux,c['clip_aux'],error_if_nonfinite=True) if aux else torch.tensor(0.)
  optimizer.step();torch.cuda.synchronize()
  row={'step':step+1,'action_loss':action,'aux_loss':auxloss,'aux_weight':weight,
       'main_gradient_norm':float(main_norm),'aux_gradient_norm':float(aux_norm),'lr':optimizer.param_groups[0]['lr'],
       'step_seconds':time.monotonic()-begin,'max_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,
       'data_indices_sha256':digest(indices),'rank':rank}
  with history.open('a') as f:f.write(json.dumps(row)+'\n')
  if rank==0 and ((step+1)%20==0 or step<3):print('UPDATE',json.dumps(row),flush=True)
  if step+1 in (1,a.stop) or (step+1)%c['save_interval']==0:
   cp=state.save(run/'checkpoints',model,optimizer,step+1,fp,rank,world)
   if rank==0:
    print('FULL_CHECKPOINT',cp,flush=True)
    export_policy(model,run,step+1,fp)
   dist.barrier()
 # A killed job after its last full save can rerun with start==stop; export still completes.
 if rank==0:
  export_policy(model,run,a.stop,fp)
  write(run/'FINISHED.json',{'passed':True,'step':a.stop,'fingerprint':fp,'finished_unix':time.time(),
     'mode':c['mode'],'evaluation_complete':False})
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
