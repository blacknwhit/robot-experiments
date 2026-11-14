"""FOCAL block visibility executed with native-GQA FlashAttention 2.

Context is evaluated once. Semantic tokens never enter the context, query or
action key sets. Packing indices are built once outside checkpointed layers.
"""
from dataclasses import dataclass
import torch
from flash_attn import flash_attn_varlen_func


@dataclass
class Block:
    q: torch.Tensor
    kv: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    max_q: int
    max_kv: int
    causal: bool = False


def make_layout(cv, sv, action_length=50):
    b, c = cv.shape
    s = sv.shape[1]
    p = c + 16
    total = p + s + action_length
    # One host read per batch; zero host synchronizations inside the 18 layers.
    valid_c = cv.detach().cpu().tolist()
    valid_s = sv.detach().cpu().tolist()
    blocks = []
    for branch in ['context', 'geometry', 'motion', 'semantic', 'action']:
        qi, ki, qlens, klens = [], [], [], []
        for i in range(b):
            context = [j for j, flag in enumerate(valid_c[i]) if flag]
            geometry, motion = list(range(c, c+8)), list(range(c+8, p))
            semantic = [p+j for j, flag in enumerate(valid_s[i]) if flag]
            action = list(range(p+s, total))
            q, kv = {'context': (context, context), 'geometry': (geometry, context+geometry),
                     'motion': (motion, context+motion), 'semantic': (semantic, context+semantic),
                     'action': (action, context+geometry+motion+action)}[branch]
            qi.extend(i*total+j for j in q); ki.extend(i*total+j for j in kv)
            qlens.append(len(q)); klens.append(len(kv))
        if not sum(qlens):
            continue
        device = cv.device
        lengths_q = torch.tensor([0]+qlens, dtype=torch.int32, device=device)
        lengths_k = torch.tensor([0]+klens, dtype=torch.int32, device=device)
        blocks.append(Block(torch.tensor(qi,device=device),torch.tensor(ki,device=device),
                            lengths_q.cumsum(0,dtype=torch.int32),lengths_k.cumsum(0,dtype=torch.int32),
                            max(qlens),max(klens),branch=='semantic'))
    # Q index sets are disjoint. Invalid padding outputs are never consumed as KV
    # or supervised tokens, so their attention outputs can be zero.
    return blocks


def focal_flash(q, k, v, blocks, scale):
    b, h, length, d = q.shape
    q = q.transpose(1,2).reshape(b*length,h,d)
    k = k.transpose(1,2).reshape(b*length,k.shape[1],d)
    v = v.transpose(1,2).reshape(b*length,v.shape[1],d)
    outputs, indices = [], []
    for block in blocks:
        result = flash_attn_varlen_func(q.index_select(0,block.q),k.index_select(0,block.kv),
                                       v.index_select(0,block.kv),block.cu_q,block.cu_kv,
                                       block.max_q,block.max_kv,dropout_p=0.,softmax_scale=scale,
                                       causal=block.causal,deterministic=True)
        outputs.append(result);indices.append(block.q)
    out=q.new_zeros(b*length,h,d).index_copy(0,torch.cat(indices),torch.cat(outputs))
    return out.view(b,length,h*d)
