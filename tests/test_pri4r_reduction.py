"""Four CPU ranks verify global valid-XYZ normalization against a serial oracle.

This exercises distributed mathematical reduction, not four-GPU NCCL or capacity.
"""
from pathlib import Path
from contextlib import nullcontext
import os,sys,socket,unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import activate
activate('pri4r_paper',overlay=False)
from openpi.models_pytorch.point_track import point_track_loss_terms,distributed_point_track_loss


def rank_check(rank,world,url):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=world)
    try:
        g=torch.Generator().manual_seed(987)
        x=torch.randn(128,8,3,3,generator=g);target=torch.randn(128,8,3,3,generator=g)
        valid=torch.rand(128,8,3,generator=g)>.25
        valid[:20]=False;target[~valid]=float('nan')
        for accum in (1,2):
            torch.manual_seed(123);model=DDP(torch.nn.Linear(3,3,bias=False))
            reference=torch.nn.Linear(3,3,bias=False);reference.load_state_dict(model.module.state_dict())
            ref_sum,ref_n,_=point_track_loss_terms(reference(x),target,valid)
            (ref_sum/ref_n).backward()
            per_rank=128//world;micro=per_rank//accum
            selected=slice(rank*per_rank,(rank+1)*per_rank)
            count=valid[selected].sum().float()*3;dist.all_reduce(count)
            for j in range(accum):
                part=slice(rank*per_rank+j*micro,rank*per_rank+(j+1)*micro)
                with (model.no_sync() if j<accum-1 else nullcontext()):
                    total,_,_=point_track_loss_terms(model(x[part]),target[part],valid[part])
                    loss=distributed_point_track_loss(total,count,world_size=world,accumulation_steps=accum)
                    (loss/accum).backward()  # Accelerator's accumulation scaling.
            torch.testing.assert_close(model.module.weight.grad,reference.weight.grad,rtol=2e-5,atol=2e-7)
            del model
        dist.barrier()
    finally:dist.destroy_process_group()


class DistributedReduction(unittest.TestCase):
    def test_four_rank_unequal_validity_and_accumulation(self):
        # Shared cluster filesystems may not implement FileStore's advisory
        # locking. These local CPU ranks communicate over loopback instead.
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0))
            port=reservation.getsockname()[1]
        mp.spawn(rank_check,args=(4,f'tcp://127.0.0.1:{port}'),nprocs=4,join=True)


if __name__=='__main__':unittest.main()
