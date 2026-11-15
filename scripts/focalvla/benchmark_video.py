"""Measure real planned RGB decoding/prefetch without generating teacher caches."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'runtimes/focalvla'))
from focalvla.common import read_json, atomic_json
from focalvla.data import CacheDataset


class VideoOnlyDataset(CacheDataset):
    def _sample(self,index,images):
        # Use the production reader and batched fetch, omitting teacher/state IO.
        return np.stack(images)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prefetch-factor',type=int,default=4)
    parser.add_argument('--batches',type=int,default=96)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--skip-steps',type=int,default=80)
    parser.add_argument('--output')
    args=parser.parse_args()
    if min(args.workers,args.batches,args.batch_size,args.prefetch_factor)<1 or args.skip_steps<0:
        parser.error('workers, batches and batch-size must be positive; skip-steps must be nonnegative')
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(1)
    resources=read_json(args.resources)
    draws=np.load(Path(args.prepared)/'draws.npy',mmap_mode='r')
    warmup=max(16,args.workers*2)
    n=(warmup+args.batches)*args.batch_size
    ids=draws[args.skip_steps:].reshape(-1)[:n]
    if len(ids)!=n: parser.error('Sampling plan too short for this benchmark window')
    ids=ids.reshape(-1,args.batch_size).tolist()
    loader=DataLoader(VideoOnlyDataset(args.prepared,resources['dataset_root']),batch_sampler=ids,
                      num_workers=args.workers,prefetch_factor=args.prefetch_factor,persistent_workers=True,
                      multiprocessing_context='spawn')
    started=time.perf_counter();latencies=[]
    for i,batch in enumerate(loader,1):
        assert tuple(batch.shape)==(args.batch_size,3,256,256,3)
        if i==warmup:
            startup=time.perf_counter()-started;began=tick=time.perf_counter()
        elif i>warmup:
            now=time.perf_counter();latencies.append(now-tick);tick=now
    elapsed=time.perf_counter()-began
    report={'workers':args.workers,'prefetch_factor':args.prefetch_factor,'warmup_batches':warmup,'warmup_seconds':startup,
            'measured_batches':len(latencies),'batch_size':args.batch_size,'seconds':elapsed,
            'samples_per_second':len(latencies)*args.batch_size/elapsed,
            'mean_batch_seconds':elapsed/len(latencies),'p95_next_seconds':float(np.quantile(latencies,.95)),
            'scope':'Production video reader on original full-plan sample order; video decode and collation only. Not a GPU training benchmark.'}
    if args.output: atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
