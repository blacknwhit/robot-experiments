"""Ordered, process-parallel RoboCasa v3 windows for the JAX training input."""
from dataclasses import dataclass
import math
import sys
import time

import numpy as np

from data import history_indices, sample_current


@dataclass
class Episode:
    episode_id: int
    first_row: int
    length: int
    cameras: tuple  # (video path, episode start time in this video)


def _columns(base, modality):
    result = []
    for key in base.modality_keys[modality]:
        subkey = key.split(".", 1)[1]
        field = getattr(base.lerobot_modality_meta, modality)[subkey]
        absolute = getattr(base.metadata.modalities, modality)[subkey].absolute
        result.append((field.original_key or subkey, slice(field.start, field.end), absolute))
    return result


def _video_window(path, timestamps):
    import cv2
    # Use one decoder thread per DataLoader worker; parallelism is across samples.
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    else:
        cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (fps > 0 and count > 0 and np.isfinite(timestamps).all()):
            raise ValueError(f"Invalid FPS, frame count or timestamps for {path}")
        # Same nearest timestamp as abs(arange(N)/fps - ts).argmin(), lower wins ties.
        t = np.asarray(timestamps, dtype=np.float64)
        lo = np.clip(np.floor(t * fps).astype(np.int64), 0, count - 1)
        hi = np.minimum(lo + 1, count - 1)
        indices = np.where(np.abs(hi / fps - t) < np.abs(lo / fps - t), hi, lo)
        unique = np.unique(indices)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(unique[0])):
            raise OSError(f"Cannot seek {path} to {unique[0]}")
        decoded = {}
        wanted = set(unique.tolist())
        for index in range(int(unique[0]), int(unique[-1]) + 1):
            if not cap.grab():
                raise OSError(f"Cannot grab {path} frame {index}")
            if index in wanted:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise OSError(f"Cannot retrieve {path} frame {index}")
                decoded[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return [decoded[int(i)] for i in indices]
    finally:
        cap.release()


class Windows:
    """A picklable sampling catalogue. Arrow handles are lazy and per process."""
    def __init__(self, stream):
        self.cfg = stream.cfg
        self.prob = stream.mixture.dataset_sampling_weights
        self.tasks = []
        self.bundles = []
        bundle_ids = {}
        for view, probabilities in zip(stream.mixture.datasets, stream.mixture.trajectory_sampling_weights, strict=True):
            base = view._base_dataset
            if base._lerobot_version != "v3.0" or base.video_backend != "opencv" or base._action_mode != "abs" or base.transforms.transforms:
                raise ValueError("Fast path requires the accepted RoboCasa v3 OpenCV raw-data configuration")
            source = str(base.dataset_path)
            if source not in bundle_ids:
                bundle_ids[source] = len(self.bundles)
                for modality in ("video", "state", "language", "action"):
                    indices = np.arange(self.cfg["action_horizon"]) if modality == "action" else [0]
                    if any(not np.array_equal(base.delta_indices[k], indices) for k in base.modality_keys[modality]):
                        raise ValueError(f"Unsupported {modality} offsets")
                lk = base.modality_keys["language"][0]
                field = base.lerobot_modality_meta.annotation[lk.split(".", 1)[1]]
                self.bundles.append((base._v3_reader, _columns(base, "state"), _columns(base, "action"),
                                     field.original_key or lk, base.tasks["task"].to_dict()))
            episodes = []
            for eid, length in zip(view.trajectory_ids, view.trajectory_lengths, strict=True):
                meta = base.trajectory_ids_to_metadata[int(eid)]
                cameras = []
                for key in base.modality_keys["video"]:
                    subkey = key.split(".", 1)[1]
                    col = base.lerobot_modality_meta.video[subkey].original_key or subkey
                    cameras.append((sys.intern(str(base.get_video_path(int(eid), subkey))),
                                    float(meta.get("videos/from_timestamps", {}).get(col, 0.0))))
                start = int(meta["dataset_from_index"])
                if int(meta["dataset_to_index"]) - start != length:
                    raise ValueError(f"Episode length mismatch for {eid}")
                episodes.append(Episode(int(eid), start, int(length), tuple(cameras)))
            self.tasks.append((bundle_ids[source], view.dataset_name, probabilities, episodes))

    def sample(self, index, *, current=None):
        from openpi_client.image_tools import resize_with_pad
        cfg = self.cfg
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], int(index)]))
        task = int(rng.choice(len(self.tasks), p=self.prob))
        bundle, name, probabilities, episodes = self.tasks[task]
        ep = episodes[int(rng.choice(len(episodes), p=probabilities))]
        anchor = sample_current(ep.length, rng) if current is None else int(current)
        if not 0 <= anchor < ep.length:
            raise ValueError(anchor)
        times = history_indices(anchor, cfg["history_frames"], cfg["history_interval"])
        first = times[0]
        stop = min(ep.length, anchor + cfg["action_horizon"])
        reader, state_fields, action_fields, lang, task_labels = self.bundles[bundle]
        columns = list(dict.fromkeys(["timestamp", "episode_index", lang] +
                                     [c for c, _, _ in state_fields + action_fields]))
        table = reader._open_dataset().data.table
        frame = table.slice(ep.first_row + first, stop - first).select(columns).to_pandas()
        if len(frame) != stop - first or not (frame["episode_index"] == ep.episode_id).all():
            raise ValueError(f"Arrow window outside episode {ep.episode_id}")
        rows = np.asarray(times) - first
        stamps = frame["timestamp"].to_numpy()[rows]
        cameras = []
        for path, offset in ep.cameras:
            rgb = _video_window(path, stamps + offset)
            cameras.append([resize_with_pad(im, 224, 224) for im in rgb])
        arrays = {c: np.stack(frame[c].to_numpy()) for c in {c for c, _, _ in state_fields + action_fields}}
        states = np.concatenate([arrays[c][rows, sl] for c, sl, _ in state_fields], axis=1).astype(np.float32)
        labels = [str(task_labels[int(v)]) for v in frame[lang].to_numpy()[rows]]
        horizon = cfg["action_horizon"]
        valid = stop - anchor
        parts = []
        for c, sl, absolute in action_fields:
            values = arrays[c][anchor - first:, sl]
            out = np.zeros((horizon, values.shape[-1]), dtype=np.float32)
            out[:valid] = values
            if absolute and valid < horizon:
                out[valid:] = values[-1]
            parts.append(out)
        units = [dict(image=[cam[t] for cam in cameras], state=states[t], lang=label)
                 for t, label in enumerate(labels)]
        units[-1]["action"] = np.concatenate(parts, axis=1)
        return dict(index=int(index), dataset=name, task_index=task, episode=ep.episode_id,
                    length=ep.length, times=times, units=units)

    def __getitem__(self, index):
        return self.sample(index)


def _as_samples(samples):
    # NumPy uint8 IPC avoids sending fp32 images or depending on /dev/shm size.
    return samples


def _init_worker(worker_id):
    import cv2
    import pyarrow
    pyarrow.set_cpu_count(1)
    cv2.setNumThreads(1)


def batches(windows, processor, cfg, cursor, stop, *, workers=None, timing=None):
    """One DataLoader iterator per run. A global batch draws chunks from several workers."""
    from torch.utils.data import DataLoader
    size = cfg["global_batch_size"]
    workers = int(cfg["reader"]["num_workers"] if workers is None else workers)
    chunk = min(size, int(cfg["reader"].get("loader_chunk", 4)))
    if size < 1 or chunk < 1 or workers < 0 or (stop - cursor) % size or cursor < 0 or stop < cursor:
        raise ValueError("Invalid batch size, worker count, or cursor range")
    chunk = math.gcd(size, chunk)
    kwargs = {}
    if workers:
        # DataLoader counts *chunks per worker*. Size the outstanding work in
        # full host batches so decode can overlap a complete optimizer update.
        ahead = int(cfg["reader"].get("prefetch_batches", 1))
        if ahead < 1:
            raise ValueError("prefetch_batches must be positive")
        chunks_per_worker = math.ceil(ahead * (size // chunk) / workers)
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True,
                      prefetch_factor=chunks_per_worker, worker_init_fn=_init_worker)
    loader = DataLoader(windows, sampler=range(cursor, stop), batch_size=chunk,
                        num_workers=workers, collate_fn=_as_samples, pin_memory=False, **kwargs)
    expected = cursor
    pending = []
    start = time.perf_counter()
    for samples in loader:
        # DataLoader returns chunks in sampler order regardless of worker completion order.
        if [s["index"] for s in samples] != list(range(expected, expected + len(samples))):
            raise RuntimeError("DataLoader changed the global sample order")
        expected += len(samples)
        pending.extend(samples)
        if len(pending) == size:
            loaded = time.perf_counter()
            batch = processor.batch(pending)
            pending.clear()
            if timing is not None:
                timing.update(loader_wait_s=loaded-start, host_pack_s=time.perf_counter()-loaded)
            yield batch
            start = time.perf_counter()
    if pending or expected != stop:
        raise RuntimeError("Incomplete global batch")
