"""Inference-safe placeholders for OpenPI's eagerly imported training registry."""

from __future__ import annotations


DATASET_SOUP_REGISTRY = {
    "target50": [],
    "target_atomic_seen": [],
    "target_composite_seen": [],
    "target_composite_unseen": [],
    "pretrain_human300_mg60": [],
    "pretrain_human300": [],
}


def get_ds_meta(**kwargs):
    """Return inert metadata retained only inside unused training configs."""
    return {"path": "", "filter_key": None, **kwargs}

