"""Four CPU ranks verify global valid-XYZ normalization against a serial oracle.

This exercises distributed mathematical reduction, not four-GPU NCCL or capacity.
"""
from pathlib import Path
from contextlib import nullcontext
import os,sys,socket,unittest,json,tempfile
from types import SimpleNamespace
from unittest.mock import patch
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
        valid[:32]=False;target[~valid]=float('nan')  # One entire rank has no auxiliary labels.
        for accum,all_invalid in ((1,False),(2,False),(1,True),(2,True)):
            mask=torch.zeros_like(valid) if all_invalid else valid
            torch.manual_seed(123);model=DDP(torch.nn.Linear(3,3,bias=False))
            reference=torch.nn.Linear(3,3,bias=False);reference.load_state_dict(model.module.state_dict())
            ref_sum,ref_n,_=point_track_loss_terms(reference(x),target,mask)
            (ref_sum/ref_n.clamp_min(1)).backward()
            per_rank=128//world;micro=per_rank//accum
            selected=slice(rank*per_rank,(rank+1)*per_rank)
            count=mask[selected].sum().float()*3;dist.all_reduce(count)
            for j in range(accum):
                part=slice(rank*per_rank+j*micro,rank*per_rank+(j+1)*micro)
                with (model.no_sync() if j<accum-1 else nullcontext()):
                    total,_,_=point_track_loss_terms(model(x[part]),target[part],mask[part])
                    loss=distributed_point_track_loss(total,count,world_size=world,accumulation_steps=accum)
                    (loss/accum).backward()  # Accelerator's accumulation scaling.
            torch.testing.assert_close(model.module.weight.grad,reference.weight.grad,rtol=2e-5,atol=2e-7)
            del model
        # Exercise the actual engineering audit with local-zero/global-positive
        # labels, then global-zero labels with existing Adam momentum.
        from engineering_audit import native_audited_step
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__();self.policy_model=torch.nn.Module()
                self.policy_model.shared=torch.nn.Linear(3,3,bias=False)
                self.policy_model.point_track_head=torch.nn.Linear(3,3,bias=False)
            def forward(self,value):
                hidden=self.policy_model.shared(value)
                return hidden,self.policy_model.point_track_head(hidden)
        torch.manual_seed(345);model=DDP(Wrapper())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-10)
        acc=SimpleNamespace(unwrap_model=lambda m:m.module,sync_gradients=True,
            gradient_accumulation_steps=1,process_index=rank,num_processes=world,is_main_process=rank==0)
        with tempfile.TemporaryDirectory() as tmp:
            trainer=SimpleNamespace(model=model,accelerator=acc,total_batch_size=128,
                completed_steps=0,config=SimpleNamespace(output_dir=tmp))
            part=slice(rank*32,(rank+1)*32)
            def step(t,batch,_):
                mask=batch[0]['point_valid']
                count=mask.sum().float()*3;dist.all_reduce(count);t._point_global_count=count
                hidden,pred=model(x[part])
                total,_,_=point_track_loss_terms(pred,target[part],mask)
                loss=hidden.square().mean()+distributed_point_track_loss(total,count,
                    world_size=world,accumulation_steps=1)
                loss.backward();optimizer.step();optimizer.zero_grad()
                return {}
            names=['shared.weight','point_track_head.weight']
            with patch('engineering_audit.audited_parameter_names',return_value=names):
                for index,mask in enumerate((valid[part],torch.zeros_like(valid[part]))):
                    trainer.completed_steps=index
                    native_audited_step(step,trainer,[{'point_valid':mask}])
            proofs=[json.loads(line) for line in (Path(tmp)/f'native_update_rank{rank}.jsonl').read_text().splitlines()]
            if rank==0:
                first=proofs[0]['parameters']['point_track_head.weight']
                assert proofs[0]['local_valid_xyz']==0 and proofs[0]['global_valid_xyz']>0
                assert first['gradient_squared_norm']==0 and first['update_max']>0
            assert proofs[1]['global_valid_xyz']==0
            assert proofs[1]['parameters']['point_track_head.weight']['gradient_squared_norm']==0
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
