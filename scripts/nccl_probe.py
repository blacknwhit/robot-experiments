"""Run under torchrun on allocated GPUs before training (not on a login host)."""
import datetime,json,os
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
x=torch.tensor([rank+1.0],device=f'cuda:{local}');dist.all_reduce(x)
if x.item()!=world*(world+1)/2:raise RuntimeError('NCCL all_reduce mismatch')
print(json.dumps({'rank':rank,'world':world,'GPU':torch.cuda.get_device_name(local),'sum':x.item(),'passed':True}),flush=True)
dist.destroy_process_group()
