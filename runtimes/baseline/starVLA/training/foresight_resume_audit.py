"""Validate all restored master weights and Adam moments before the first update."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist

def verify_group_tensors(master, expected, actual_state, expected_state, bit16, *, padding, runtime_padding, label):
    """Audit ZeRO2's lean saved master and rigid (padded) Adam state exactly."""
    assert isinstance(padding, int) and 0 <= padding <= master.numel(), label
    assert padding == runtime_padding, f'Checkpoint/runtime alignment padding differs: {label}'
    assert master.ndim == expected.ndim == bit16.ndim == 1, label
    assert master.numel() == expected.numel() + padding, f'Unexpected master extent: {label}'
    assert bit16.numel() == master.numel(), f'BF16 partition extent differs: {label}'
    # DeepSpeed state_dict strips ONLY the master padding in rigid checkpoints;
    # native load_state_dict restores it as zeros. It keeps full Adam moments.
    if padding:
        assert bool((master.detach()[expected.numel():] == 0).all()), f'Nonzero master padding: {label}'
        assert bool((bit16.detach()[expected.numel():] == 0).all()), f'Nonzero BF16 padding: {label}'
    checked = 0
    pairs = [(master[:expected.numel()], expected)]
    for key in ('exp_avg', 'exp_avg_sq'):
        actual, reference = actual_state[key], expected_state[key]
        assert actual.numel() == reference.numel() == master.numel(), f'Adam extent differs: {label} {key}'
        pairs.append((actual, reference))
    for actual, reference in pairs:
        for start in range(0, actual.numel(), 4*1024*1024):
            value = actual.detach().reshape(-1)[start:start+4*1024*1024].cpu()
            saved = reference.reshape(-1)[start:start+4*1024*1024]
            assert torch.equal(value, saved), f'Restore mismatch {label}'
            assert bool(torch.isfinite(value).all()), f'Nonfinite restored state {label}'
            checked += value.numel()
    for start in range(0, expected.numel(), 4*1024*1024):
        actual = bit16.detach()[start:start+4*1024*1024].cpu()
        # Do not include a padding element in the last BF16 slice.
        actual = actual[:min(4*1024*1024, expected.numel()-start)]
        reference = expected[start:start+4*1024*1024].to(actual.dtype)
        assert torch.equal(actual, reference), f'BF16 model/master mismatch {label}'
    return checked

def verify_loaded_partition(engine, checkpoint, output_dir):
    rank=dist.get_rank()
    source=Path(checkpoint)
    manifest_path=source/'migration_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'target_world_size': dist.get_world_size(), 'completed_steps': engine.global_steps, 'rng': {str(rank): {'new_cuda_seed': None}}}
    saved=torch.load(source/f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
                     map_location='cpu',weights_only=False,mmap=True)['optimizer_state_dict']
    opt=engine.optimizer
    assert dist.get_world_size()==manifest['target_world_size']
    assert engine.global_steps==manifest['completed_steps']
    rng=torch.load(source/f'random_states_{rank}.pkl',map_location='cpu',weights_only=False)
    assert random.getstate()==rng['random_state'], f'Python RNG restore failed: rank={rank}'
    current_numpy=np.random.get_state(); saved_numpy=rng['numpy_random_seed']
    assert current_numpy[0]==saved_numpy[0] and np.array_equal(current_numpy[1],saved_numpy[1]) and current_numpy[2:]==saved_numpy[2:]
    assert torch.equal(torch.get_rng_state(),rng['torch_manual_seed']), f'CPU RNG restore failed: rank={rank}'
    assert torch.equal(torch.cuda.get_rng_state(),rng['torch_cuda_manual_seed'][rank]), f'CUDA RNG restore failed: rank={rank}'
    checked=0
    groups = len(opt.single_partition_of_fp32_groups)
    assert len(saved['single_partition_of_fp32_groups']) == len(saved['group_paddings']) == len(opt.groups_padding) == groups
    assert saved['partition_count'] == list(opt.partition_count) == [dist.get_world_size()] * groups
    for i,master in enumerate(opt.single_partition_of_fp32_groups):
        expected=saved['single_partition_of_fp32_groups'][i]
        actual_state=opt.optimizer.state[opt.optimizer.param_groups[i]['params'][0]]
        expected_state=saved['base_optimizer_state']['state'][i]
        assert actual_state['step'].item()==expected_state['step'].item()==manifest['completed_steps']
        assert opt.optimizer.param_groups[i]['lr']==saved['base_optimizer_state']['param_groups'][i]['lr']
        bit16=opt.parallel_partitioned_bit16_groups[i][rank]
        checked += verify_group_tensors(master, expected, actual_state, expected_state, bit16,
                    padding=saved['group_paddings'][i], runtime_padding=opt.groups_padding[i],
                    label=f'rank={rank} group={i}')
    seed=manifest['rng'][str(rank)]['new_cuda_seed']
    if seed is not None:
        torch.cuda.manual_seed(seed)
    report={'rank':rank,'completed_steps':engine.global_steps,'fp32_values_checked':checked,
            'master_and_adam_moments_bitwise_equal':True,'bf16_model_matches_master':True,
            'rng_restore_verified_before_child_seed':True,'new_cuda_seed':seed,
            'zero_alignment_padding_by_group':saved['group_paddings'],
            'nonparameter_padding_verified_zero':True}
    (Path(output_dir)/f'resume_partition_audit_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FORESIGHT_4GPU_RESTORE_AUDIT_OK',json.dumps(report),flush=True)
    dist.barrier()
