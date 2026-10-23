"""Masked cumulative track supervision; diagnostics never reweight training."""
import torch


def validate_horizons(horizons, length):
    horizons = tuple(horizons)
    if not horizons or any(type(h) is not int or not 1 <= h <= length for h in horizons):
        raise ValueError('Cumulative horizons must be positive integers within the track')
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError('Cumulative horizons must be sorted and unique')
    return horizons


def trajectory_loss_terms(pred, target, valid, horizons=(10, 25, 50)):
    """Per-horizon FP32 L1 sums / h and valid XYZ counts, before DDP reduction.

    Endpoint h requires every increment 0:h to be valid. Missing intermediate
    states cannot be crossed; targets are never clamped or extrapolated at EOF.
    """
    if pred.ndim != 4 or pred.shape != target.shape or pred.shape[-1] != 3:
        raise ValueError('Expected equal [B,N,H,3] predictions and labels')
    if valid.dtype != torch.bool or valid.shape != pred.shape[:-1]:
        raise ValueError('Expected bool [B,N,H] validity')
    horizons = validate_horizons(horizons, pred.shape[2])
    if not torch.isfinite(pred).all() or not torch.where(valid[..., None], torch.isfinite(target), True).all():
        raise FloatingPointError('Nonfinite predictions or valid point labels')
    # Mask before subtraction: an invalid NaN must neither enter the sum nor
    # poison gradients through a later multiplication by zero.
    error = torch.where(valid[..., None], pred.float(), 0) - torch.where(valid[..., None], target.detach().float(), 0)
    cumulative = error.cumsum(dim=2)
    path_valid = valid.to(torch.int32).cumprod(dim=2).bool()
    sums, counts = [], []
    for h in horizons:
        mask = path_valid[:, :, h - 1]
        sums.append(torch.where(mask[..., None], cumulative[:, :, h - 1].abs() / h, 0).sum())
        counts.append(mask.sum().to(torch.float32) * 3)
    return torch.stack(sums), torch.stack(counts)


def auxiliary_ramp(completed_steps, warmup_steps):
    if completed_steps < 0 or warmup_steps < 0:
        raise ValueError('Negative optimizer clock or auxiliary warmup')
    return 1.0 if warmup_steps == 0 else min(1.0, (completed_steps + 1) / warmup_steps)


def diagnostic_names(horizons=(10, 25, 50)):
    names = []
    for group in ['robot', 'object', 'object_le_0p1mm', 'object_gt_1mm']:
        names += [f'track_diag/adj_{group}_{k}' for k in ['abs_sum', 'zero_abs_sum', 'xyz_count']]
    for h in horizons:
        for group in ['robot', 'object', 'object_path_le_10mm', 'object_path_gt_10mm']:
            names += [f'track_diag/fde{h}_{group}_{k}' for k in ['l2_sum', 'zero_l2_sum', 'point_count']]
    return names


@torch.no_grad()
def point_diagnostic_sums(pred, target, valid, categories, horizons=(10, 25, 50)):
    """Global-reducible diagnostic numerators/denominators, meters, detached.

    Per-step low/high-motion labels and per-horizon GT path-length labels are
    reporting groups only. None enters the training objective or model inputs.
    """
    if categories.shape != pred.shape[:2] or not ((categories == 1) | (categories == 2)).all():
        raise ValueError('Expected category1=robot,2=object for each input point')
    horizons = validate_horizons(horizons, pred.shape[2])
    pred = pred.detach().float()
    target = torch.where(valid[..., None], target.detach().float(), 0)
    err = torch.where(valid[..., None], pred - target, 0)
    robot = (categories == 1)[:, :, None]
    objects = (categories == 2)[:, :, None]
    motion = target.norm(dim=-1)
    groups = [valid & robot, valid & objects, valid & objects & (motion <= .0001), valid & objects & (motion > .001)]
    values = []
    for mask in groups:
        values += [(err.abs() * mask[..., None]).sum(), (target.abs() * mask[..., None]).sum(), mask.sum().float() * 3]
    path_valid = valid.to(torch.int32).cumprod(dim=2).bool()
    cumulative_error = err.cumsum(dim=2)
    cumulative_target = target.cumsum(dim=2)
    path_length = motion.cumsum(dim=2)
    for h in horizons:
        ok = path_valid[:, :, h - 1]
        low = path_length[:, :, h - 1] <= .01
        groups = [ok & robot[:, :, 0], ok & objects[:, :, 0], ok & objects[:, :, 0] & low, ok & objects[:, :, 0] & ~low]
        fde, zero = cumulative_error[:, :, h - 1].norm(dim=-1), cumulative_target[:, :, h - 1].norm(dim=-1)
        for mask in groups:
            values += [(fde * mask).sum(), (zero * mask).sum(), mask.sum().float()]
    return torch.stack(values)
