"""Read immutable episode tracks, without importing a simulator or loading futures as inputs.

Positions are stored once in world coordinates. A training window uses the inverse
of its *first* base pose for all 51 positions, never the future robot poses. Only
``point_current`` is an auxiliary-head input; delta/valid are loss-only fields.
"""

from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os

import numpy as np

SCHEMA = "pri4r_world_tracks_v1"


def episode_key(source_prefix, source_episode_index):
    prefix = str(source_prefix).strip("/")
    if not prefix or ".." in Path(prefix).parts or "\\" in prefix or ":" in prefix:
        raise ValueError("source_prefix must be a relative canonical source prefix")
    index = int(source_episode_index)
    if index < 0:
        raise ValueError("negative source episode")
    return f"{prefix}/episode_{index:06d}"


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("invalid anchor pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("pose is not homogeneous")
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(r), 1, atol=2e-5):
        raise ValueError("anchor rotation is not SO(3)")
    return pose


def timestamp_tolerance(timestamps, frame_period, minimum=1e-5, source_dtype=None):
    raw = np.asarray(timestamps)
    dtype = np.dtype(source_dtype or raw.dtype)
    if not np.issubdtype(dtype, np.floating) or not raw.size or not np.isfinite(raw).all():
        raise ValueError("timestamps must be nonempty finite floating-point values")
    quantum = float(abs(np.spacing(np.array(np.max(np.abs(raw)), dtype=dtype))))
    tolerance = max(float(minimum), 2 * quantum)
    if tolerance >= .1 * frame_period:
        raise ValueError("timestamp precision insufficient to distinguish control steps")
    return tolerance


def point_window(world_tracks, base_poses, timestamps, start, *, horizon=50,
                 frame_period=0.05, time_tolerance=1e-5, frame_valid=None,
                 expected_timestamp=None, output_kind="adjacent_displacement"):
    """Return finite arrays with exactly N,H,3 deltas and no EOF clamping.

    Invalid targets are zero-filled and masked. A missing future state does not
    remove the sample from the original action/Query training distribution.
    Endpoint validity is per point, and temporal gaps invalidate the respective
    adjacent displacement. The first point must be valid for every target.
    """
    if output_kind not in ("adjacent_displacement", "start_relative_position"):
        raise ValueError("Unsupported point output semantics")
    start, horizon = int(start), int(horizon)
    if horizon < 1 or not (0 <= start < len(world_tracks)):
        raise ValueError("invalid window index/horizon")
    if world_tracks.ndim != 3 or world_tracks.shape[-1] != 3:
        raise ValueError("world_tracks must be [T,N,3]")
    if len(base_poses) != len(world_tracks) or len(timestamps) != len(world_tracks):
        raise ValueError("track/pose/time length mismatch")
    anchor = validate_pose(base_poses[start])
    stop = min(len(world_tracks), start + horizon + 1)
    # Only the current small slice is inspected; a dtype conversion must not
    # cause false timestamp gaps in long float32 episodes.
    time_tolerance = timestamp_tolerance(timestamps[start:stop], frame_period, time_tolerance)
    if expected_timestamp is not None and not np.isclose(
            float(timestamps[start]), float(expected_timestamp), atol=time_tolerance, rtol=0):
        raise ValueError("training timestamp differs from cached GT timestamp")
    # Copy only this 51-frame slice; memmap does not decompress whole episodes.
    xyz = np.asarray(world_tracks[start:stop], dtype=np.float64)
    ts = np.asarray(timestamps[start:stop], dtype=np.float64)
    finite = np.isfinite(xyz).all(-1) & np.isfinite(ts)[:, None]
    if frame_valid is not None:
        supplied = np.asarray(frame_valid[start:stop], dtype=bool)
        if supplied.ndim == 1:
            supplied = supplied[:, None]
        finite &= np.broadcast_to(supplied, finite.shape)
    current_valid = finite[0].copy()
    safe = np.where(finite[..., None], xyz, 0.0)
    # Row-vector inverse rigid transform: (p_world - t_world) @ R_world_from_base.
    local = (safe - anchor[:3, 3]) @ anchor[:3, :3]
    point_current = np.where(current_valid[:, None], local[0], 0.0).astype(np.float32)
    n = world_tracks.shape[1]
    delta = np.zeros((n, horizon, 3), dtype=np.float32)
    valid = np.zeros((n, horizon), dtype=bool)
    steps = len(xyz) - 1
    if steps:
        aligned = np.isclose(ts, ts[0] + np.arange(len(ts)) * frame_period,
                             atol=time_tolerance, rtol=0)
        contiguous = np.isclose(np.diff(ts), frame_period, atol=time_tolerance, rtol=0)
        times_ok = contiguous & aligned[1:] & aligned[:-1]
        good = finite[1:] & finite[:-1] & current_valid[None, :] & times_ok[:, None]
        differences = (local[1:] - local[:-1]).transpose(1, 0, 2)
        valid[:, :steps] = good.T
        delta[:, :steps] = np.where(good.T[..., None], differences, 0).astype(np.float32)
    result = {"point_current": point_current, "point_delta": delta,
            "point_valid": valid, "point_current_valid": current_valid,
            "point_meta": {"schema": SCHEMA, "frame_index": start, "horizon": horizon,
                           "num_points": n, "coordinate_frame": "window_start_robot_base", "units": "meters",
                           "output_kind": output_kind, "available_endpoints": len(xyz)}}
    if output_kind == "start_relative_position":
        # Read actual endpoints in the SAME frozen frame; do not integrate
        # zero-filled invalid deltas and silently label those positions as real.
        position_valid = np.logical_and.accumulate(valid, axis=1)
        positions = np.zeros_like(delta)
        if steps:
            relative = (local[1:] - local[0:1]).transpose(1, 0, 2)
            positions[:, :steps] = np.where(position_valid[:, :steps, None], relative, 0).astype(np.float32)
        result.pop("point_delta")
        result.update(point_position=positions, point_valid=position_valid)
        result["point_meta"]["position_validity"] = "complete_prefix_v1"
    return result


def validate_episode(path, *, expected_cache_key=None, verify_hashes=False):
    path = Path(path)
    meta = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("complete") is not True:
        raise ValueError(f"incomplete/unsupported GT: {path}")
    if expected_cache_key is not None and meta.get("cache_key") != expected_cache_key:
        raise ValueError("GT cache identity mismatch")
    shape = (int(meta["frames"]), int(meta["points"]), 3)
    specs = {"world_tracks.npy": (shape, np.dtype("float32")),
             "base_poses.npy": ((shape[0], 4, 4), np.dtype("float64")),
             "timestamps.npy": ((shape[0],), np.dtype("float64")),
             "point_geom_ids.npy": ((shape[1],), np.dtype("int32")),
             "point_local.npy": ((shape[1], 3), np.dtype("float64")),
             "point_face_ids.npy": ((shape[1],), np.dtype("int32")),
             "point_barycentric.npy": ((shape[1], 3), np.dtype("float64"))}
    arrays = {}
    for name, (expected_shape, dtype) in specs.items():
        file = path / name
        a = np.load(file, mmap_mode="r", allow_pickle=False)
        if a.shape != expected_shape or a.dtype != dtype:
            raise ValueError(f"GT shape/dtype mismatch: {name}")
        if int(meta["files"][name]["bytes"]) != file.stat().st_size:
            raise ValueError(f"GT file size mismatch: {name}")
        if verify_hashes and digest_file(file) != meta["files"][name]["sha256"]:
            raise ValueError(f"GT file hash mismatch: {name}")
        arrays[name] = a
    if "point_categories.npy" in meta["files"]:
        file = path / "point_categories.npy"
        a = np.load(file, mmap_mode="r", allow_pickle=False)
        if a.shape != (shape[1],) or a.dtype not in (np.dtype("int8"), np.dtype("uint8")) or not np.isin(a, [1, 2]).all():
            raise ValueError("Invalid arm/object diagnostic categories")
        if digest_file(file) != meta["files"]["point_categories.npy"]["sha256"]:
            raise ValueError("Diagnostic category SHA mismatch")
        arrays["point_categories.npy"] = a
    return meta, arrays


class PointTrackStore:
    """A bounded LRU of readonly memmaps, reset automatically after worker fork."""
    def __init__(self, root, *, cache_size=8, expected_config_sha256=None,
                 num_points=1024, horizon=50, verify_hashes=False,
                 output_kind="adjacent_displacement"):
        self.root = Path(root)
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.expected_config_sha256 = expected_config_sha256
        self.num_points, self.horizon = int(num_points), int(horizon)
        self.verify_hashes = bool(verify_hashes)
        if output_kind not in ("adjacent_displacement", "start_relative_position"):
            raise ValueError("Unsupported point output semantics")
        self.output_kind = output_kind
        self._pid, self._cache = os.getpid(), OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    @staticmethod
    def _close_arrays(arrays):
        for a in arrays.values():
            if getattr(a, "_mmap", None) is not None:
                a._mmap.close()

    def close(self):
        for _, arrays in self._cache.values():
            self._close_arrays(arrays)
        self._cache.clear()

    def _open(self, key):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        meta, arrays = validate_episode(self.root / key, verify_hashes=self.verify_hashes)
        if meta["episode_key"] != key or int(meta["points"]) != self.num_points:
            self._close_arrays(arrays)
            raise ValueError("GT source identity/point count mismatch")
        if self.expected_config_sha256 and meta["config_sha256"] != self.expected_config_sha256:
            self._close_arrays(arrays)
            raise ValueError("GT sampling configuration mismatch")
        self._cache[key] = (meta, arrays)
        while len(self._cache) > self.cache_size:
            _, (_, evicted) = self._cache.popitem(last=False)
            self._close_arrays(evicted)
        return meta, arrays

    def read(self, source_prefix, source_episode_index, frame_index, *, timestamp=None):
        key = episode_key(source_prefix, source_episode_index)
        meta, arrays = self._open(key)
        result = point_window(arrays["world_tracks.npy"], arrays["base_poses.npy"],
                              arrays["timestamps.npy"], frame_index, horizon=self.horizon,
                              frame_period=1.0 / meta["fps"],
                              time_tolerance=meta.get("time_tolerance", 1e-5),
                              expected_timestamp=timestamp, output_kind=self.output_kind)
        if "point_categories.npy" in arrays:
            result["point_categories"] = np.asarray(arrays["point_categories.npy"], dtype=np.int8).copy()
        result["point_meta"].update(episode_key=key, cache_key=meta["cache_key"],
                                    config_sha256=meta["config_sha256"])
        return result
