"""New-process strict full-state load, joint/cache parity, point-free actions."""
import pyarrow  # noqa: F401
import argparse
import json
from pathlib import Path
import sys
from common import activate, POSITION_VARIANTS, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import yaml
    c = yaml.safe_load(args.config.read_text())
    variant = c["portable"]["variant"]
    if variant not in POSITION_VARIANTS:
        raise ValueError("V3 engineering probe only")
    activate(variant)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.openpi.pi05 import POLICY_RNG_SEED_KEY
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from safetensors import safe_open
    from strict_state_loader import strict_full_state_load
    c["framework"]["pi05"]["init_checkpoint"] = None
    model = build_framework(OmegaConf.create(c))
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as f:
        wrapped = all(k.startswith("policy_model.") for k in f.keys())
    proof = strict_full_state_load(model if wrapped else model.policy_model, args.checkpoint)
    model.to("cuda:0").eval()  # Preserve the accepted selective precision.
    # Only our own locally generated engineering sample is loaded.
    sample = torch.load(args.sample, weights_only=False, map_location="cpu")
    policy = model.policy_model
    if not policy.paligemma_with_expert.separate_prefix_attention:
        raise AssertionError("Required cache-aligned prefix path disabled")
    preprocess = policy._preprocess_observation
    policy._preprocess_observation = lambda obs, train=True: preprocess(obs, train=False)
    captured = {}
    def capture(module, inputs, output):
        captured["velocity"] = output["velocity"].detach()
    handle = policy.register_forward_hook(capture)
    parity = []
    try:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=policy.action_in_proj.weight.dtype == torch.bfloat16)
        with torch.no_grad(), autocast:
            obs, truth, _ = model._prepare_batch([sample], include_actions=True)
            truth = truth.float()
            for seed in (71, 72):
                noise = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed)).to("cuda")
                t = torch.tensor([.5], device="cuda")
                model([sample], noise=noise, time=t)
                joint = captured["velocity"]
                images, masks, tokens, token_masks, state = preprocess(obs, train=False)
                prefix, pad, att = policy.embed_prefix(images, masks, tokens, token_masks)
                pos = torch.cumsum(pad, dim=1) - 1
                mask = policy._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
                _, cache = policy.paligemma_with_expert.forward(
                    attention_mask=mask, position_ids=pos, past_key_values=None,
                    inputs_embeds=[prefix, None], use_cache=True)
                cached = policy.denoise_step(state, pad, cache,
                                            t[:, None, None] * noise + (1 - t[:, None, None]) * truth, t)
                delta = cached.float() - joint.float()
                relative = float(delta.norm() / joint.float().norm().clamp_min(1e-12))
                if not torch.isfinite(cached).all() or relative >= 1e-5:
                    raise AssertionError(f"Joint/cache mismatch: {relative}")
                parity.append(dict(seed=seed, relative_l2=relative, max_abs=float(delta.abs().max())))
    finally:
        handle.remove()
        policy._preprocess_observation = preprocess
    sample = {k: v for k, v in sample.items() if not k.startswith("point_")}
    sample[POLICY_RNG_SEED_KEY] = 7
    def forbidden(*args, **kwargs):
        raise AssertionError("Training-only point/Embedding branch executed at inference")
    policy.point_track_head.register_forward_pre_hook(forbidden)
    policy.point_track_head.embedding.register_forward_pre_hook(forbidden)
    x = model.predict_action([sample])["actions"]
    y = model.predict_action([sample])["actions"]
    if x.shape != (1, 50, 12) or not np.isfinite(x).all() or not np.array_equal(x, y):
        raise AssertionError("Action shape/finite/deterministic request contract failed")
    report = dict(passed=True, variant=variant, checkpoint_sha256=sha256(args.checkpoint),
                  strict_state=proof, joint_cache_parity=parity, actions_shape=list(x.shape),
                  point_inputs_required=False, point_head_called=False, finite=True,
                  deterministic_requests=True, gpu=torch.cuda.get_device_name(0),
                  engineering_only=True, closed_loop_benchmark=False)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
