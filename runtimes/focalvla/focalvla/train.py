"""torchrun entry: full-data FOCAL training, from base or exact continuation."""
import argparse
import contextlib
import datetime
import gc
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .common import read_json, atomic_json, digest, sha256, source_digest, lr_at
from . import checkpoint as ckpt
from .data import CacheDataset, PlanBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resources', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stop-after', type=int, help='Bounded verification; never changes the LR schedule or save interval.')
    args = parser.parse_args()
    config, resources = read_json(args.config), read_json(args.resources)
    root, out = Path(args.prepared), Path(args.output)
    rank, local, world = (int(os.environ.get(k, '0' if k != 'WORLD_SIZE' else '1')) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    assert world == config['world_size'], (world, config['world_size'])
    batch, accumulation = config['microbatch_per_rank'], config['gradient_accumulation']
    assert world*batch*accumulation == config['global_batch']
    torch.cuda.set_device(local); torch.set_num_threads(4)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        import fcntl
        lock = (out/'writer.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dist.barrier()
    ready = read_json(root/'READY.json')
    assert ready['plan_contract'] == read_json(root/'PLAN.json')['contract']
    assert ready['total_steps'] == config['total_steps'] and ready['global_batch'] == config['global_batch']
    assert ready['episodes'] == config['expected_episodes'] and ready['tasks'] == config['expected_tasks']
    assert ready['missing_episodes'] == 0 and ready['invalid_labels'] == 0
    assert ready['norm_sha256'] == sha256(Path(resources['norm_assets'])/'norm_stats.json')
    assert ready['tokenizer_sha256'] == sha256(resources['tokenizer'])
    for filename, expected in ready['training_files'].items():
        assert sha256(root/filename) == expected, filename
    stats = read_json(root/'target_stats.json')
    draws = np.load(root/'draws.npy', mmap_mode='r')
    validity = np.load(root/'validity.npy', mmap_mode='r')
    resource_identity = [None]
    if rank == 0:
        resource_identity[0] = {'base_sha256': sha256(resources['pi05_weights'])}
    dist.broadcast_object_list(resource_identity)
    protocol = {'config': config, 'prepared_sha256': sha256(root/'READY.json'),
                'source_sha256': source_digest(), **resource_identity[0]}
    contract = digest(protocol)
    if rank == 0:
        if (out/'PROTOCOL.json').exists():
            assert read_json(out/'PROTOCOL.json')['contract'] == contract, 'Output belongs to another run.'
        atomic_json(out/'PROTOCOL.json', dict(protocol, contract=contract))
        atomic_json(out/'resources.resolved.json', resources)
    start, file = ckpt.latest(out/'checkpoints', contract)
    if start == config['total_steps']:
        if rank == 0: print('Training already complete.', flush=True)
        dist.destroy_process_group(); return
    from .model import FocalPolicy
    from .adapter import Collate
    ckpt.seed(config['seed'])
    raw = FocalPolicy(resources, stats, load_base=file is None).cuda()
    # This head is unused by flow matching; freezing avoids unused DDP parameters.
    raw.native.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad],
        lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'],
        weight_decay=config['weight_decay'], fused=True)
    if file:
        start = ckpt.restore(file, raw, optimizer, contract, rank, world, draws,
                            config['global_batch'], lambda n: lr_at(n, config))
        gc.collect(); torch.cuda.empty_cache()
    else:
        ckpt.seed(config['seed'] + rank)
    model = DDP(raw, device_ids=[local], output_device=local, find_unused_parameters=False,
                gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=64)
    stop = min(config['total_steps'], args.stop_after or config['total_steps'])
    loader_kwargs = dict(num_workers=config['num_workers'], pin_memory=True,
                         collate_fn=Collate(resources), generator=torch.Generator().manual_seed(config['seed']))
    if config['num_workers']:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=config['prefetch_factor'], multiprocessing_context='spawn')
    loader = DataLoader(CacheDataset(root, config['episode_cache_size']),
        batch_sampler=PlanBatches(root, start, stop, rank, world, batch, accumulation), **loader_kwargs)
    iterator = iter(loader)
    if rank == 0:
        # Preserve the old attempt log; truncate only progress beyond the committed checkpoint.
        metrics = out/'metrics.jsonl'
        if metrics.exists():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            old = metrics.read_text(encoding='utf-8'); (out/f'metrics.{stamp}.jsonl').write_text(old, encoding='utf-8')
            import json
            keep = []
            for line in old.splitlines():
                try:
                    if json.loads(line)['step'] <= start: keep.append(line)
                except (ValueError, KeyError): pass
            metrics.write_text(''.join(line+'\n' for line in keep), encoding='utf-8')
        print({'event':'READY','start_step':start,'world':world,'global_batch':config['global_batch'],
               'microbatch':batch,'accumulation':accumulation,'resume':str(file) if file else None}, flush=True)
    raw.train()
    loss_keys = ('action','semantic','geo','motion')
    weights = config['loss_weights']
    sums = torch.zeros(4, device=device, dtype=torch.float64)
    torch.cuda.synchronize(); began = time.perf_counter(); window = 0; wait_seconds = 0.
    for step in range(start, stop):
        optimizer.zero_grad(set_to_none=False)
        for group in optimizer.param_groups: group['lr'] = lr_at(step, config)
        counts = np.maximum(1, validity[draws[step]].sum(0))
        for micro in range(accumulation):
            load_start = time.perf_counter(); samples, targets = next(iterator)
            wait_seconds += time.perf_counter()-load_start
            factors = {'action':1/accumulation, 'semantic':1/accumulation,
                       'geo':world*float(targets['geo_valid'].sum())/counts[0],
                       'motion':world*float(targets['motion_valid'].sum())/counts[1]}
            with model.no_sync() if micro < accumulation-1 else contextlib.nullcontext():
                losses = model(samples, targets)
                loss = losses['action']*factors['action'] + sum(weights[k]*losses[k]*factors[k] for k in weights)
                loss.backward()
            sums += torch.stack([losses[k].detach().double()*factors[k] for k in loss_keys])
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step(); window += 1
        completed = step+1
        if completed % config['log_interval'] == 0 or completed == stop:
            torch.cuda.synchronize()
            duration = torch.tensor(time.perf_counter()-began, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            dist.all_reduce(sums); sums /= world*window
            values = dict(zip(loss_keys, sums.cpu().tolist()))
            values['loss'] = values['action']+sum(weights[k]*values[k] for k in weights)
            if rank == 0:
                import json
                record = {'step':completed, **values, 'lr':lr_at(step,config), 'seconds_per_step':float(duration)/window,
                          'samples_per_second':window*config['global_batch']/float(duration),
                          'rank0_data_wait_seconds_per_step':wait_seconds/window,
                          'rank0_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30, 'gradient_norm':float(norm)}
                with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
            sums.zero_(); window = 0; wait_seconds = 0.; began = time.perf_counter()
        if completed % config['save_interval'] == 0:
            dist.barrier(); states = [None]*world; dist.all_gather_object(states, ckpt.rng_state())
            if rank == 0:
                saved = ckpt.save(out/'checkpoints',raw,optimizer,completed,contract,states,draws,
                                  config['global_batch'],lr_at(completed,config))
                print({'event':'CHECKPOINT','step':completed,'file':saved.name}, flush=True)
            dist.barrier()
            began = time.perf_counter()
    if stop == config['total_steps'] and rank == 0:
        assert stop % config['save_interval'] == 0, 'Final step must be a configured save step.'
        atomic_json(out/'TRAINING_COMPLETE.json', {'steps':stop,'contract':contract,'draws':stop*config['global_batch']})
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
