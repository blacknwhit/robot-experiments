"""Read current and future RGB together without clamping future supervision."""
from __future__ import annotations

import numpy as np


def read_video_targets(path, timestamps, nominal_fps):
    """One OpenCV open; return an explicit mask rather than clamp past EOF."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if fps <= 0 or count <= 0 or min(height, width) <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        times = np.asarray(timestamps, dtype=np.float64)
        # Nearest frame, choosing the lower index for an exact half-frame tie.
        indices = np.ceil(times * fps - .5).astype(np.int64)
        present = (indices >= 0) & (indices < count) & (np.abs(indices / fps - times) <= .5 / nominal_fps + 1e-6)
        frames = np.zeros((len(times), height, width, 3), dtype=np.uint8)
        for j in np.flatnonzero(present):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[j]))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Failed to decode valid frame {indices[j]} from {path}")
            frames[j] = frame[:, :, ::-1]
        return frames, present
    finally:
        cap.release()


def resolve_targets(timestamps, current, offsets, fps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if fps <= 0 or current < 0 or current >= len(timestamps):
        raise ValueError("Invalid episode or FPS")
    if len(offsets) != 2 or offsets[0] <= 0 or offsets[0] >= offsets[1]:
        raise ValueError("Two distinct future offsets required")
    desired = timestamps[current] + np.asarray(offsets) / fps
    # RoboCasa stores image, state and action per control step. Explicitly check
    # their timestamps instead of silently assuming that t+H is a valid frame.
    indices = current + np.asarray(offsets, dtype=np.int64)
    in_episode = indices < len(timestamps)
    actual = np.full(2, -1., dtype=np.float64)
    actual[in_episode] = timestamps[indices[in_episode]]
    valid = in_episode & (np.abs(actual - desired) <= (0.5 / fps + 1e-6))
    return indices, desired, actual, valid


def read_foresight_video(dataset, episode, current):
    from openpi_client.image_tools import resize_with_pad

    cfg = dataset.data_cfg
    if dataset.video_backend != "opencv":
        raise ValueError("RoboCasa foresight timestamp validation requires the baseline OpenCV backend")
    offsets = tuple(int(x) for x in cfg.foresight_offsets)
    include_targets = bool(cfg.get("foresight_targets", False))
    views = tuple(cfg.foresight_view_keys)
    if views != tuple(dataset.modality_keys["video"]):
        raise ValueError("Explicit foresight camera mapping differs from current adapter")
    if tuple(dataset.delta_indices[dataset.modality_keys["action"][0]]) != tuple(range(offsets[-1])):
        raise ValueError("Future endpoint does not match action chunk")
    times = dataset.curr_traj_data["timestamp"].to_numpy()
    fps = float(dataset.lerobot_info_meta["fps"])
    indices, desired, actual, time_valid = resolve_targets(times, current, offsets, fps)
    targets = np.zeros((2, len(views), 3, 224, 224), dtype=np.float32) if include_targets else None
    valid = np.zeros((2, len(views)), dtype=bool)
    current_valid = np.zeros(len(views), dtype=bool)
    raw = {}
    mapping = []
    for v, key in enumerate(views):
        short = key.removeprefix("video.")
        original = dataset.lerobot_modality_meta.video[short].original_key
        path = dataset.get_video_path(episode, short)
        mapping.append({"slot": v, "source": original, "adapter_key": key})
        if not path.exists():
            raw[key] = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            continue
        meta = dataset.trajectory_ids_to_metadata[episode]
        origin = float(meta.get("videos/from_timestamps", {}).get(original, 0.))
        selected = [current] + (indices[time_valid].tolist() if include_targets else [])
        frames, present = read_video_targets(path, times[selected] + origin, fps)
        raw[key] = frames[:1]
        current_valid[v] = present[0]
        if not include_targets:
            continue
        for j, k in enumerate(np.flatnonzero(time_valid)):
            if not present[j + 1]:
                continue
            frame = resize_with_pad(np.asarray(frames[j + 1]), 224, 224)
            targets[k, v] = np.transpose(frame.astype(np.float32) / 127.5 - 1., (2, 0, 1))
            valid[k, v] = True
    raw["_foresight_targets"] = {
        "current_valid": current_valid,
        "sample_meta": {"dataset": dataset.dataset_name, "episode": int(episode),
            "current_frame": int(current), "target_frames": indices.tolist(),
            "current_timestamp": float(times[current]), "desired_timestamps": desired.tolist(),
            "actual_timestamps": actual.tolist(), "fps": fps, "offsets": list(offsets),
            "camera_order": list(views), "camera_mapping": mapping},
    }
    if include_targets:
        raw["_foresight_targets"].update(future_images=targets, future_valid=valid)
    return raw
