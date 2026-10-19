"""OpenPI PyTorch augmentation with explicit, temporally shared parameters.

Matches preprocessing_pytorch.py at upstream 215abfb217db, including local-batch
randomness and interpolation. The rotation grid is expanded over the batch (the
upstream implementation constructs a singleton grid). Camera roles are taken
from the RoboCasa source keys, not its compatibility slot names.
"""
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

MODE = "official_pytorch_temporal_sync_v1"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass(frozen=True)
class AugmentParams:
    top: int
    left: int
    crop: bool
    angle: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    saturation: torch.Tensor


def sample_params(device, *, wrist, height=224, width=224):
    """One draw per local batch/camera, with the upstream draw order/ranges."""
    top = left = 0
    angle = torch.zeros(1, device=device)
    if not wrist:
        max_h, max_w = height - int(height * .95), width - int(width * .95)
        if max_h > 0 and max_w > 0:
            top = int(torch.randint(0, max_h + 1, (1,), device=device))
            left = int(torch.randint(0, max_w + 1, (1,), device=device))
        angle = torch.rand(1, device=device) * 10 - 5
    return AugmentParams(top, left, not wrist, angle,
                         .7 + torch.rand(1, device=device) * .6,
                         .6 + torch.rand(1, device=device) * .8,
                         .5 + torch.rand(1, device=device))


def apply_params(images, params):
    """[N,3,H,W], [-1,1], float32; all statistics are per image."""
    if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.float32:
        raise ValueError("Expected float32 RGB [N,3,H,W]")
    # Preserve the upstream channels-last reduction order for numeric checks.
    value = images.permute(0, 2, 3, 1) / 2.0 + .5
    height, width = value.shape[1:3]
    if params.crop:
        ch, cw = int(height * .95), int(width * .95)
        value = value[:, params.top:params.top + ch, params.left:params.left + cw, :]
        value = F.interpolate(value.permute(0, 3, 1, 2), size=(height, width),
                              mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        if torch.abs(params.angle) > .1:
            angle = params.angle * torch.pi / 180.0
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height, device=value.device),
                                          torch.linspace(-1, 1, width, device=value.device), indexing="ij")
            gx = grid_x * torch.cos(angle) - grid_y * torch.sin(angle)
            gy = grid_x * torch.sin(angle) + grid_y * torch.cos(angle)
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(value.shape[0], -1, -1, -1)
            value = F.grid_sample(value.permute(0, 3, 1, 2), grid,
                                  mode="bilinear", padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)
    value = value * params.brightness
    mean = value.mean(dim=[1, 2, 3], keepdim=True)
    value = (value - mean) * params.contrast + mean
    gray = value.mean(dim=-1, keepdim=True)
    value = gray + (value - gray) * params.saturation
    return (torch.clamp(value, 0, 1) * 2.0 - 1.0).permute(0, 3, 1, 2).contiguous()


def preprocess_pair(observation, future_images, *, camera_order, train=True, params=None):
    """Return a fresh observation and targets; never mutate loader input arrays.

    Future images only enter the teacher branch. Current image preprocessing
    uses neither future pixels nor their statistics. Passing params is useful
    for numerical acceptance without coupling the action-noise RNG stream.
    """
    if tuple(camera_order) != CAMERAS or tuple(observation.images) != SLOTS:
        raise ValueError("Unexpected RoboCasa camera/compatibility slot mapping")
    b = observation.state.shape[0]
    if future_images.shape != (b, 2, 3, 3, 224, 224):
        raise ValueError("Expected future images [B,2,3,3,224,224]")
    if any(image.shape != (b, 3, 224, 224) for image in observation.images.values()):
        raise ValueError("Current images must have passed the standard 224 RGB input transform")
    if not torch.isfinite(future_images).all():
        raise FloatingPointError("Nonfinite future RGB")
    if not train:
        return observation, future_images, None
    if params is not None and len(params) != 3:
        raise ValueError("Exactly one augmentation parameter set per camera required")
    out_images, out_future, used = {}, [], []
    # No autocast: augment RGB in FP32, before either encoder's precision policy.
    with torch.autocast(future_images.device.type, enabled=False):
        for v, slot in enumerate(SLOTS):
            p = sample_params(future_images.device, wrist=(v == 1)) if params is None else params[v]
            current = observation.images[slot].float()
            temporal = future_images[:, :, v].reshape(b * 2, 3, 224, 224).float()
            transformed = apply_params(torch.cat([current, temporal], dim=0), p)
            out_images[slot] = transformed[:b]
            out_future.append(transformed[b:].reshape(b, 2, 3, 224, 224))
            used.append(p)
    return replace(observation, images=out_images), torch.stack(out_future, dim=2), tuple(used)


def preprocess_current_camera_roles(observation, *, train=True, params=None):
    """Use the exact FIXED current-image augmentation, without future RGB input."""
    if tuple(observation.images) != SLOTS:
        raise ValueError("Unexpected RoboCasa compatibility slot mapping")
    b = observation.state.shape[0]
    if any(im.shape != (b, 3, 224, 224) for im in observation.images.values()):
        raise ValueError("Expected already transformed224 RGB inputs")
    if not train:
        return observation, None
    if params is not None and len(params) != 3:
        raise ValueError("Exactly three camera parameter sets required")
    used, out = [], {}
    device = next(iter(observation.images.values())).device
    with torch.autocast(device.type, enabled=False):
        for i, slot in enumerate(SLOTS):
            p = sample_params(device, wrist=(i == 1)) if params is None else params[i]
            out[slot] = apply_params(observation.images[slot].float(), p)
            used.append(p)
    return replace(observation, images=out), tuple(used)
