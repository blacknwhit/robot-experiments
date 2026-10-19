"""Strict loading of a complete native state dictionary, including tied aliases.

Native training exports serialize every state_dict key. safetensors.load_model
instead expects redundant names of a tied parameter to have been omitted.
Keep every native key, reject conflicting aliases, and verify every loaded value.
"""
from collections import defaultdict
import json
import torch
from safetensors.torch import load_file


def strict_full_state_load(model, filename):
    expected = model.state_dict()
    saved = load_file(filename, device="cpu")
    if set(saved) != set(expected):
        raise RuntimeError("Incomplete or foreign checkpoint: " + json.dumps({
            "missing": sorted(set(expected) - set(saved)),
            "unexpected": sorted(set(saved) - set(expected)),
        }))
    for name, value in saved.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise RuntimeError("Checkpoint tensor shape mismatch: " + name)
    groups = defaultdict(list)
    for name, value in model.named_parameters(remove_duplicate=False):
        groups[id(value)].append(name)
    for name, value in model.named_buffers(remove_duplicate=False):
        if name in saved:
            groups[id(value)].append(name)
    aliases = [sorted(names) for names in groups.values() if len(names) > 1]
    for names in aliases:
        first = saved[names[0]]
        for name in names[1:]:
            if first.dtype != saved[name].dtype or not torch.equal(first, saved[name]):
                raise RuntimeError("Conflicting serialized tied weights: " + repr(names))
    result = model.load_state_dict(saved, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loaded = model.state_dict()
    for name, value in saved.items():
        actual = loaded[name].detach().cpu()
        if not torch.equal(actual, value.to(dtype=actual.dtype)):
            raise RuntimeError("Loaded checkpoint tensor differs: " + name)
    proof = {"strict": True, "keys": len(saved), "all_keys_present": True,
             "all_loaded_values_exact_at_model_dtype": True,
             "serialized_tied_aliases_exact": aliases,
             "checkpoint_keys_dropped": 0}
    print("STRICT_NATIVE_FULL_STATE_LOADED " + json.dumps(proof), flush=True)
    return proof
