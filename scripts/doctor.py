"""Explicit import/hardware diagnostics, executed in the selected policy environment."""
import argparse,json,sys
from common import activate,VARIANTS
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--cuda',action='store_true');a=p.parse_args()
    activate(a.variant)
    import pyarrow,torch,transformers
    from starVLA.training import train_starvla
    from openpi.training import config
    result={'python':sys.version,'torch':str(torch.__version__),'transformers':transformers.__version__,'pyarrow':pyarrow.__version__,
            'openpi_config':config.get_config('pi05_pretrain_human300').name,'cuda':torch.cuda.is_available()}
    if a.cuda:
        if not torch.cuda.is_available():raise RuntimeError('GPU not exposed to this process')
        result['GPUs']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        result['NCCL']=torch.cuda.nccl.version()
        for i in range(torch.cuda.device_count()):
            x=torch.ones((128,128),device=f'cuda:{i}');y=x@x
            if not torch.isfinite(y).all():raise RuntimeError('CUDA math failed')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
