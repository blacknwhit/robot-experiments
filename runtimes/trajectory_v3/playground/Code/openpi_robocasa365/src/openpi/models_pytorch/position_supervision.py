"""Dense position + adjacent-difference L1, with globally reducible counts."""
import torch

from openpi.models_pytorch.trajectory_supervision import validate_horizons


def difference_validity(valid):
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[-1] < 1:
        raise ValueError("Expected bool [B,N,H] position validity")
    previous = torch.cat((torch.ones_like(valid[..., :1]), valid[..., :-1]), dim=-1)
    return valid & previous


def adjacent(values):
    return values - torch.cat((torch.zeros_like(values[:, :, :1]), values[:, :, :-1]), dim=2)


def position_loss_terms(pred, target, valid):
    if pred.ndim != 4 or pred.shape != target.shape or pred.shape[-1] != 3:
        raise ValueError("Expected matching [B,N,H,3] relative positions")
    diff_valid = difference_validity(valid)
    if valid.shape != pred.shape[:-1]:
        raise ValueError("Position validity shape mismatch")
    # The reader promises a complete prefix; reject accidental post-gap supervision.
    if not torch.equal(valid, valid.to(torch.int32).cumprod(dim=2).bool()):
        raise ValueError("V3 position mask must be a complete valid prefix")
    if not torch.isfinite(pred).all() or not torch.where(valid[..., None], torch.isfinite(target), True).all():
        raise FloatingPointError("Nonfinite prediction or valid position target")
    # Sanitize BEFORE arithmetic; invalid NaN * 0 is still NaN.
    safe_target = torch.where(valid[..., None], target.detach().float(), 0)
    safe_pred = torch.where(valid[..., None], pred.float(), 0)
    pos_sum = (safe_pred - safe_target).abs().sum()
    diff_error = adjacent(safe_pred) - adjacent(safe_target)
    diff_sum = torch.where(diff_valid[..., None], diff_error.abs(), 0).sum()
    return pos_sum, valid.sum().float() * 3, diff_sum, diff_valid.sum().float() * 3


def diagnostic_names(horizons=(10, 25, 50)):
    result = []
    for metric in ("adj", "ade"):
        for group in ("robot", "object"):
            result += [f"position_diag/{metric}_{group}_{k}" for k in ("sum", "zero_sum", "count")]
    for h in horizons:
        for group in ("robot", "object", "object_path_le_10mm", "object_path_gt_10mm"):
            result += [f"position_diag/fde{h}_{group}_{k}" for k in ("sum", "zero_sum", "count")]
    return result


@torch.no_grad()
def position_diagnostic_sums(pred, target, valid, categories, horizons=(10, 25, 50)):
    """Adjacent coordinate MAE and ADE/FDE L2 in meters, never training weights."""
    validate_horizons(horizons, pred.shape[2])
    if categories.shape != pred.shape[:2] or not ((categories == 1) | (categories == 2)).all():
        raise ValueError("Expected robot=1/object=2 diagnostic categories")
    y = torch.where(valid[..., None], target.detach().float(), 0)
    p = torch.where(valid[..., None], pred.detach().float(), 0)
    dy, dp = adjacent(y), adjacent(p)
    dv = difference_validity(valid)
    result = []
    robot, objects = (categories == 1), (categories == 2)
    for metric in ("adj", "ade"):
        for group in (robot, objects):
            mask = (dv if metric == "adj" else valid) & group[..., None]
            if metric == "adj":
                err, zero, mult = (dp - dy).abs().sum(-1), dy.abs().sum(-1), 3
            else:
                err, zero, mult = (p - y).norm(dim=-1), y.norm(dim=-1), 1
            result += [torch.where(mask, err, 0).sum(), torch.where(mask, zero, 0).sum(),
                       mask.sum().float() * mult]
    path_length = dy.norm(dim=-1).cumsum(dim=2)
    for h in horizons:
        low = path_length[:, :, h - 1] <= .01
        for group in (robot, objects, objects & low, objects & ~low):
            mask = valid[:, :, h - 1] & group
            result += [torch.where(mask, (p[:, :, h - 1] - y[:, :, h - 1]).norm(dim=-1), 0).sum(),
                       torch.where(mask, y[:, :, h - 1].norm(dim=-1), 0).sum(), mask.sum().float()]
    return torch.stack(result)
