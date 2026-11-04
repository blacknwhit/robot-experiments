"""Same auxiliary predictor for NULL, matching and shuffled action conditions."""
import torch
from torch import nn
from torch.nn import functional as F

class CausalActionEncoder(nn.Module):
    def __init__(self,dim=12,width=64):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(dim,width),nn.GELU())
        self.recurrent=nn.GRU(width,width,batch_first=True)
    def forward(self,actions):
        return self.recurrent(self.input(actions.float()))[0]

class ActionConditionedHead(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.action_encoder=CausalActionEncoder()
        self.action_projection=nn.Linear(64,base.fusion_in.out_features)
        nn.init.zeros_(self.action_projection.weight);nn.init.zeros_(self.action_projection.bias)
    def forward(self,inputs,actions):
        hidden,valid,points=inputs[:3]
        if len(inputs)>3:points=torch.where(inputs[3][...,None],points,0)
        context=self.base.encode_time(hidden,valid)
        encoded=self.base.point_mlp(points.to(self.base.point_mlp[0].weight.dtype))
        width=context.shape[-1]
        pp=F.linear(encoded,self.base.fusion_in.weight[:,:width])
        tp=F.linear(context,self.base.fusion_in.weight[:,width:],self.base.fusion_in.bias)
        ap=self.action_projection(self.action_encoder(actions))
        output=[]
        for start in range(0,points.shape[1],self.base.config.point_chunk_size):
            joint=pp[:,start:start+self.base.config.point_chunk_size,None]+(tp+ap)[:,None]
            output.append(self.base.fusion_out(F.gelu(joint)))
        return torch.cat(output,dim=1)

def objective(pred,target,valid):
    safe=torch.where(valid[...,None],target.float(),0)
    error=torch.where(valid[...,None],pred.float()-safe,0)
    adjacent=error.abs().sum()/(valid.sum()*3).clamp_min(1)
    integrated=error.cumsum(2);prefix=valid.long().cumprod(2).bool();terms=[]
    for h in (10,25,50):
        mask=prefix[:,:,h-1]
        terms.append((integrated[:,:,h-1].abs()*mask[...,None]).sum()/(mask.sum()*3).clamp_min(1)/h)
    return 4*(adjacent+torch.stack(terms).mean())
