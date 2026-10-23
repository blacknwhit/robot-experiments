import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        
        num_workers = int(vla_dataset_cfg.get("num_workers", 16))
        group_sampling = bool(vla_dataset_cfg.get("group_sampling", False))
        group_size = int(vla_dataset_cfg.get("group_size", 0))
        per_device_batch_size = int(vla_dataset_cfg.per_device_batch_size)
        if group_sampling:
            if group_size <= 0:
                raise ValueError("group_size must be positive")
            if per_device_batch_size % group_size:
                pi05_cfg = cfg.framework.get("pi05", {})
                stateless_foresight = (
                    cfg.framework.name == "OpenPIPi05"
                    and bool(pi05_cfg.get("foresight", {}).get("enabled", False))
                    and not bool(pi05_cfg.get("memory", {}).get("enabled", False))
                    and not bool(pi05_cfg.get("wan_condition", {}).get("enabled", False))
                )
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                can_split = (
                    bool(vla_dataset_cfg.get("allow_group_split_across_ranks", False))
                    and stateless_foresight
                    and group_size % per_device_batch_size == 0
                    and (per_device_batch_size * world_size) % group_size == 0
                )
                if not can_split:
                    raise ValueError(
                        "Ordered groups must fit a per-device batch, or explicitly use "
                        "stateless foresight with complete groups in each global batch. "
                        f"group_size={group_size}, batch={per_device_batch_size}, world={world_size}"
                    )
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=bool(vla_dataset_cfg.get("pin_memory", True)),
            persistent_workers=num_workers > 0,
            prefetch_factor=int(vla_dataset_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
            drop_last=group_sampling,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
