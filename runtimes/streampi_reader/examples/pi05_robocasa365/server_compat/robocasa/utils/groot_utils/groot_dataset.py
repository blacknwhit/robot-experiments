"""Type-compatible placeholders never instantiated during policy inference."""

from __future__ import annotations

from dataclasses import dataclass


LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"


@dataclass
class ModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]


class LeRobotSingleDataset:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("RoboCasa training datasets are unavailable in the inference-only OpenPI server")


class LeRobotMixtureDataset:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("RoboCasa training datasets are unavailable in the inference-only OpenPI server")

