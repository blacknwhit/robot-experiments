"""CPU-only offline GT replay; resumable immutable episode outputs and sharding.

CLI: python -m starVLA.pri4r.generate --manifest input.json --config config.json
     --output ROOT --workers 4 --shard-index 0 --num-shards 1

Manifest: {"repo": ..., "revision": <pinned commit>, "episodes": [{
  "source_prefix": "pretrain/atomic/.../YYYYMMDD", "source_episode_index": 0,
  "episode_dir": "/.../lerobot/extras/episode_000000",
  "timestamps_path": "/.../timestamps.npy", "length": 234, "fps": 20}]}

The timestamps must come from the corresponding training/source parquet. They
are not inferred from action horizon, episode length, or a guessed FPS origin.
Geometry uses recorded simulator states, never reintegrated actions. The data
alignment audit must establish these states match the training frame ordering.
"""
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from .geometry import geom_triangles, sample_surfaces, transform_points
from .reader import SCHEMA, PHYSICAL_CLOCK_POLICY, PointTrackStore, digest_file, episode_key, timestamp_tolerance, validate_episode, validate_pose

GENERATOR_VERSION = "rigid_surface_replay_physical_clock_v2"
CLOCK_POLICY = "lerobot_frame_index_with_known_initial_capture_gap_v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


@lru_cache(maxsize=32768)
def _asset_hash(path, size, mtime_ns):
    return digest_file(path)


def asset_hash(path):
    stat = Path(path).stat()
    return _asset_hash(str(path), stat.st_size, stat.st_mtime_ns)


def validate_alignment(entry, source_hashes):
    """Require exact source/current row correspondence from offline preparation."""
    proof = entry.get("alignment_evidence", {})
    if proof.get("schema") != "pri4r_source_alignment_v1" or proof.get("passed") is not True:
        raise ValueError("missing verified source/current alignment receipt")
    for name in ("source_prefix", "source_episode_index", "length"):
        if proof.get(name) != entry[name]:
            raise ValueError(f"alignment source identity mismatch: {name}")
    if not proof.get("current_dataset_name") or not isinstance(proof.get("current_episode_index"), int):
        raise ValueError("alignment receipt omits current training identity")
    if not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("source_parquet_sha256", ""))):
        raise ValueError("alignment source parquet hash missing")
    for name in ("timestamp", "frame_index", "action", "observation.state"):
        pair = proof.get("column_hashes", {}).get(name, {})
        if not re.fullmatch(r"[0-9a-f]{64}", str(pair.get("source", ""))) or pair.get("source") != pair.get("current"):
            raise ValueError(f"source/current training column mismatch: {name}")
    for name in ("states.npz", "model.xml.gz", "ep_meta.json"):
        if proof.get("source_files", {}).get(name) != source_hashes[name]:
            raise ValueError(f"alignment receipt differs from actual source: {name}")
    return proof


def validate_recorded_clocks(simulator_times, dataset_times, fps, dataset_tolerance):
    """Validate distinct simulator and synthetic LeRobot clocks without shifting rows.

    RoboCasa exports observation i by restoring states[i], while LeRobot.add_frame
    synthesizes timestamp=i/fps. The audited four episodes have one initial 2*dt
    simulator interval, then exactly dt throughout; the initial state is recorded
    separately by DataCollectionWrapper. Only this observed leading-gap pattern
    and a fully regular clock are accepted. Internal gaps, reversals, other first
    gaps, and nonfinite clocks fail; no interpolation/rescaling is performed.
    """
    sim = np.asarray(simulator_times, dtype=np.float64)
    dataset = np.asarray(dataset_times, dtype=np.float64)
    if sim.ndim != 1 or sim.shape != dataset.shape or len(sim) < 1:
        raise ValueError("simulator/dataset clock shape mismatch")
    if not np.isfinite(sim).all() or not np.isfinite(dataset).all() or fps <= 0:
        raise ValueError("nonfinite or invalid recorded clocks")
    period = 1.0 / float(fps)
    if not np.allclose(dataset, np.arange(len(dataset)) * period, atol=dataset_tolerance, rtol=0):
        raise ValueError("LeRobot timestamps differ from recorded frame_index/fps")
    intervals = np.diff(sim)
    physical_tolerance = max(1e-8, float(abs(np.spacing(np.max(np.abs(sim))))) * 8)
    if np.any(intervals <= 0):
        raise ValueError("simulator clock is not strictly increasing")
    if len(intervals):
        first = float(intervals[0])
        allowed_first = np.isclose(first, period, atol=physical_tolerance, rtol=0) or np.isclose(
            first, 2 * period, atol=physical_tolerance, rtol=0)
        if not allowed_first:
            raise ValueError("unknown initial simulator clock interval")
        if not np.allclose(intervals[1:], period, atol=physical_tolerance, rtol=0):
            raise ValueError("unexpected interior simulator clock gap/cadence")
        extra = first - period
        if abs(extra) <= physical_tolerance:
            extra = 0.0
    else:
        first, extra = None, 0.0
    return {"policy": CLOCK_POLICY, "dataset_clock": "frame_index/fps", "fps": float(fps),
            "nominal_period_seconds": period, "simulator_initial_time": float(sim[0]),
            "first_interval_seconds": first, "initial_extra_dt_seconds": float(extra),
            "physical_span_seconds": float(sim[-1] - sim[0]),
            "nominal_span_seconds": float((len(sim) - 1) * period),
            "physical_tolerance_seconds": physical_tolerance, "dataset_tolerance_seconds": float(dataset_tolerance),
            "frame_mapping": "states[i]_and_observation[i]_unchanged", "interpolation": False,
            "delta_rescaling": False, "non_nominal_interval_indices": [0] if extra else [],
            "provenance": {
                "robocasa_exporter_sha256": "43cc54b2b94a794455317d5e9d1177927977b0a29205f01a23414717b3fd1450",
                "exporter_behavior": "extract_trajectory restores states[i]; add_frame receives observation[i]/action[i] without timestamp",
                "lerobot_add_frame_behavior": "timestamp = frame_index / self.fps when timestamp argument is None",
                "collection_behavior": "DataCollectionWrapper saves initial state separately before appending post-step states"}}


@dataclass(frozen=True)
class SamplingConfig:
    # Cube size/base-body name are explicit local conventions, not unpublished
    # author values. Pin the accepted configuration before full generation.
    cube_half_extent_m: float
    base_body_name: str = "mobilebase0_base"
    num_points: int = 1024
    seed: int = 42
    geom_groups: tuple = (1,)
    primitive_segments: int = 32
    primitive_rings: int = 16
    asset_root: str = "/UNCONFIGURED_ROBOT_EXPERIMENTS_HOME"
    time_tolerance: float = 1e-5
    clock_policy: str = PHYSICAL_CLOCK_POLICY

    def __post_init__(self):
        if self.clock_policy != PHYSICAL_CLOCK_POLICY:
            raise ValueError("New paper GT requires recorded physical clock supervision")
        if self.cube_half_extent_m <= 0 or self.num_points < 1:
            raise ValueError("positive cube half-extent/point count required")
        if self.primitive_rings % 2:
            raise ValueError("primitive rings must be even")
        if not self.geom_groups or any(int(g) not in (0, 1) for g in self.geom_groups):
            raise ValueError("sampling debug/target geom groups is forbidden")


@contextmanager
def episode_lock(path):
    """Nonblocking OS lock; crashes release it without unsafe stale-PID guesses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def rewrite_assets(xml_bytes, asset_root):
    """Rewrite known package assets under the allowed root; reject escaped paths."""
    root = Path(asset_root).resolve()
    xml = ET.fromstring(xml_bytes)
    assets = []
    for element in xml.iter():
        old = element.get("file")
        if not old:
            continue
        replacement = None
        for package, project in (("robocasa", "robocasa365"), ("robosuite", "robosuite")):
            marker = f"/{package}/models/assets/"
            if marker in old:
                replacement = root / "playground/Code" / project / package / "models/assets" / old.split(marker, 1)[1]
                break
        if replacement is None:
            # Explicit absolute files already under the allowed root are accepted.
            replacement = Path(old)
        replacement = replacement.resolve()
        try:
            replacement.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"asset path outside allowed root: {old}") from exc
        if not replacement.is_file():
            raise FileNotFoundError(replacement)
        element.set("file", str(replacement))
        assets.append(str(replacement))
    # Absolute paths above make the old remote meshdir irrelevant.
    for compiler in xml.findall("compiler"):
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    return ET.tostring(xml, encoding="unicode"), sorted(set(assets))


def restore_state(model, data, flat):
    """Robosuite MjSimState flatten contract: time, qpos, qvel, act."""
    import mujoco
    flat = np.asarray(flat, dtype=np.float64)
    expected = 1 + model.nq + model.nv + model.na
    if flat.shape != (expected,) or not np.isfinite(flat).all():
        raise ValueError(f"recorded state shape/finite mismatch: {flat.shape}, expected {expected}")
    data.time = flat[0]
    offset = 1
    data.qpos[:] = flat[offset:offset + model.nq]
    offset += model.nq
    data.qvel[:] = flat[offset:offset + model.nv]
    offset += model.nv
    if model.na:
        data.act[:] = flat[offset:]
    # All body/geom poses depend only on qpos. Forward kinematics reproduces
    # mj_forward's poses without collision detection/constraint solving/rendering.
    # Numerical equivalence is checked by the real-MuJoCo integration test.
    mujoco.mj_kinematics(model, data)


def body_pose(data, body_id):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return validate_pose(pose)


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def generate_episode(entry, config, output_root, *, repo, revision):
    """Generate one episode; a pre-existing mismatched cache is never overwritten."""
    started = time.monotonic()
    cfg = config if isinstance(config, SamplingConfig) else SamplingConfig(**config)
    config_dict = asdict(cfg)
    config_dict["geom_groups"] = list(cfg.geom_groups)
    config_digest = canonical_hash(config_dict)
    key = episode_key(entry["source_prefix"], entry["source_episode_index"])
    dest = Path(output_root) / key
    source_dir = Path(entry["episode_dir"])
    sources = {"states.npz": source_dir / "states.npz", "model.xml.gz": source_dir / "model.xml.gz",
               "ep_meta.json": source_dir / "ep_meta.json", "timestamps.npy": Path(entry["timestamps_path"])}
    allowed_root = Path(cfg.asset_root).resolve()
    for path in [Path(output_root), *sources.values()]:
        try:
            path.resolve().relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(f"GT file operation outside configured allowed root: {path}") from exc
    source_hashes = {name: digest_file(path) for name, path in sources.items()}
    alignment_receipt = validate_alignment(entry, source_hashes)
    expected = entry.get("source_sha256", {})
    for name, digest in expected.items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    import mujoco
    xml_bytes = gzip.decompress(sources["model.xml.gz"].read_bytes())
    rewritten, assets = rewrite_assets(xml_bytes, cfg.asset_root)
    asset_sha256 = {p: asset_hash(p) for p in assets}
    identity = {"schema": SCHEMA, "generator": GENERATOR_VERSION, "repo": repo, "revision": revision,
                "episode_key": key, "config_sha256": config_digest, "source_sha256": source_hashes}
    identity["mujoco_version"] = mujoco.__version__
    identity["asset_content_sha256"] = canonical_hash(asset_sha256)
    identity["alignment"] = {"length": int(entry["length"]), "fps": float(entry["fps"]),
                              "timestamp_source_dtype": entry.get("timestamp_source_dtype", "from_npy")}
    identity["alignment_evidence_sha256"] = canonical_hash(alignment_receipt)
    # Generator source hashes invalidate old caches even if a developer forgets
    # to bump the human-readable version after changing coordinate/sampling code.
    identity["implementation_sha256"] = {
        p.name: digest_file(p) for p in (Path(__file__), Path(__file__).with_name("geometry.py"),
                                       Path(__file__).with_name("reader.py"))}
    cache_key = canonical_hash(identity)
    lock = Path(output_root) / ".locks" / (hashlib.sha256(key.encode()).hexdigest() + ".lock")
    with episode_lock(lock):
        if dest.exists():
            meta, cached_arrays = validate_episode(dest, expected_cache_key=cache_key, verify_hashes=True)
            PointTrackStore._close_arrays(cached_arrays)
            return {"episode_key": key, "status": "cached", "cache_key": cache_key,
                    "frames": meta["frames"], "output": str(dest)}
        model = mujoco.MjModel.from_xml_string(rewritten)
        data = mujoco.MjData(model)
        with np.load(sources["states.npz"], allow_pickle=False) as archive:
            states = archive["states"]
        raw_timestamps = np.load(sources["timestamps.npy"], allow_pickle=False)
        timestamps = raw_timestamps.astype(np.float64)
        fps = float(entry["fps"])
        if fps <= 0 or timestamps.shape != (len(states),) or len(states) != int(entry["length"]):
            raise ValueError("source state/training length or FPS mismatch")
        # Dataset timestamps are float32 in the source parquet. At long episode
        # times a fixed 1e-5 tolerance would create false holes. Bound comparison
        # by their declared source quantization, still far below one control step.
        source_time_dtype = np.dtype(entry.get("timestamp_source_dtype", raw_timestamps.dtype))
        tolerance = timestamp_tolerance(timestamps, 1 / fps, cfg.time_tolerance, source_time_dtype)
        if not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), 1 / fps, atol=tolerance, rtol=0):
            raise ValueError("noncontiguous training timestamps; do not silently repair data")
        # Recorded frame identity, proven by the mandatory exact alignment
        # receipt, is distinct from the simulator's physical time coordinate.
        clock_diagnostic = validate_recorded_clocks(states[:, 0], timestamps, fps, tolerance)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        if body_id < 1:
            raise ValueError("configured moving robot base body missing; static XML fallback forbidden")
        restore_state(model, data, states[0])
        first_base = body_pose(data, body_id)
        selected = np.flatnonzero(np.isin(model.geom_group, cfg.geom_groups) & (model.geom_rgba[:, 3] > 0))
        if not len(selected):
            raise ValueError("no physical rendering geoms in configured groups")
        rotations = np.asarray(data.geom_xmat).reshape(-1, 3, 3).copy()
        positions = np.asarray(data.geom_xpos).copy()
        geometries = {int(g): geom_triangles(model, int(g), primitive_segments=cfg.primitive_segments,
                                           primitive_rings=cfg.primitive_rings,
                                           plane_extent=cfg.cube_half_extent_m * 4 + 2)
                      for g in selected}
        episode_seed = int.from_bytes(hashlib.sha256((key + ":" + str(cfg.seed)).encode()).digest()[:8], "little")
        points = sample_surfaces(geometries, rotations, positions, first_base, count=cfg.num_points,
                                 half_extent=cfg.cube_half_extent_m, seed=episode_seed)
        del geometries
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(dest.name + ".incomplete-" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            tracks = np.lib.format.open_memmap(staging / "world_tracks.npy", mode="w+", dtype=np.float32,
                                              shape=(len(states), cfg.num_points, 3))
            poses = np.lib.format.open_memmap(staging / "base_poses.npy", mode="w+", dtype=np.float64,
                                             shape=(len(states), 4, 4))
            gids = points["geom_ids"]
            first_world = None
            max_displacement = np.zeros(cfg.num_points, dtype=np.float64)
            max_adjacent_displacement = 0.0
            previous_world = None
            for frame, state in enumerate(states):
                if frame:
                    restore_state(model, data, state)
                tracks[frame] = transform_points(points["local"], data.geom_xmat[gids].reshape(-1, 3, 3), data.geom_xpos[gids])
                poses[frame] = body_pose(data, body_id)
                if not np.isfinite(tracks[frame]).all():
                    raise ValueError(f"nonfinite point tracks at frame {frame}")
                world = np.asarray(tracks[frame], dtype=np.float64)
                if first_world is None:
                    first_world = world.copy()
                max_displacement = np.maximum(max_displacement, np.linalg.norm(world - first_world, axis=-1))
                if previous_world is not None:
                    max_adjacent_displacement = max(max_adjacent_displacement,
                                                     float(np.max(np.linalg.norm(world - previous_world, axis=-1))))
                previous_world = world
            tracks.flush()
            poses.flush()
            del tracks, poses
            np.save(staging / "timestamps.npy", timestamps)
            np.save(staging / "simulator_timestamps.npy", np.asarray(states[:, 0], dtype=np.float64))
            for source, filename, dtype in (("geom_ids", "point_geom_ids.npy", np.int32),
                                            ("local", "point_local.npy", np.float64),
                                            ("face_ids", "point_face_ids.npy", np.int32),
                                            ("barycentric", "point_barycentric.npy", np.float64)):
                np.save(staging / filename, np.asarray(points[source], dtype=dtype))
            files = {p.name: {"sha256": digest_file(p), "bytes": p.stat().st_size} for p in staging.glob("*.npy")}
            point_names = {str(int(g)): {"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)),
                                        "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])),
                                        "count": int(np.count_nonzero(gids == g))} for g in np.unique(gids)}
            meta = {**identity, "complete": True, "cache_key": cache_key,
                    "config": config_dict, "frames": len(states), "points": cfg.num_points,
                    "fps": fps, "time_tolerance": tolerance, "timestamp_source_dtype": str(source_time_dtype), "files": files,
                    "mujoco_version": mujoco.__version__, "asset_paths": assets,
                    "asset_sha256": asset_sha256,
                    "asset_path_set_sha256": canonical_hash(assets),
                    "source_state_format": "time_qpos_qvel_act", "state_to_frame": "same_index_no_integration",
                    "clock_diagnostic": clock_diagnostic,
                    "alignment_evidence": alignment_receipt,
                    "units": "metres", "stored_frame": "world", "read_frame": "selected_by_training_recipe",
                    "clock_policy": cfg.clock_policy,
                    "base_body_id": body_id, "base_body_name": cfg.base_body_name,
                    "sampling": {k: points[k] for k in ("surface_area_m2", "clipped_triangles", "selected_geom_count")},
                    "coverage_diagnostic": {"moving_points_over_1mm": int(np.count_nonzero(max_displacement > .001)),
                                             "max_displacement_m": float(max_displacement.max()),
                                             "max_adjacent_displacement_m": max_adjacent_displacement,
                                             "sampling_used_future_motion": False},
                    "point_geom_coverage": point_names, "elapsed_seconds": time.monotonic() - started}
            _atomic_json(staging / "complete.json", meta)
            _, staged_arrays = validate_episode(staging, expected_cache_key=cache_key, verify_hashes=True)
            PointTrackStore._close_arrays(staged_arrays)
            # The lock and no-overwrite check make a completed episode immutable.
            if dest.exists():
                raise FileExistsError(dest)
            os.replace(staging, dest)
        except BaseException:
            # Leave an explicitly incomplete directory for forensic inspection;
            # readers only resolve the exact final episode directory.
            raise
    return {"episode_key": key, "status": "generated", "cache_key": cache_key,
            "frames": len(states), "elapsed_seconds": time.monotonic() - started, "output": str(dest)}


def _job(arguments):
    entry, config, output, repo, revision = arguments
    if entry.get("prepare_source") is True:
        from .source_prepare import prepare_episode
        entry = prepare_episode(entry, repo=repo, revision=revision)
    return generate_episode(entry, config, output, repo=repo, revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid worker/shard configuration")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    config = asdict(SamplingConfig(**json.loads(Path(args.config).read_text(encoding="utf-8"))))
    if not manifest.get("repo") or len(manifest.get("revision", "")) != 40:
        raise ValueError("manifest requires source repository and pinned 40-character revision")
    entries = manifest["episodes"]
    keys = [episode_key(e["source_prefix"], e["source_episode_index"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source episodes in manifest")
    selected = [entry for entry, key in zip(entries, keys)
                if int(hashlib.sha256(key.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_digest = digest_file(args.manifest)
    report_path = output / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.json"
    # A separate shard lock prevents two drivers racing the summary file.
    with episode_lock(output / ".locks" / f"shard_{args.shard_index}_of_{args.num_shards}.lock"):
        jobs = [(e, config, str(output), manifest["repo"], manifest["revision"]) for e in selected]
        results, errors = [], []
        run_id = uuid.uuid4().hex
        last_summary = time.monotonic()
        common = {"manifest_sha256": manifest_digest, "config_sha256": canonical_hash(config),
                  "shard_index": args.shard_index, "num_shards": args.num_shards,
                  "expected_episodes": len(selected), "run_id": run_id}
        journal_path = report_path.with_suffix(".jsonl")
        _atomic_json(report_path, {**common, "complete": False, "completed_episodes": 0, "failed_episodes": 0})
        with journal_path.open("a", encoding="utf-8") as journal, ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    entry = futures[future]
                    result = {"episode_key": episode_key(entry["source_prefix"], entry["source_episode_index"]),
                              "type": type(exc).__name__, "error": str(exc)}
                    errors.append(result)
                print(json.dumps(result, ensure_ascii=True), flush=True)
                journal.write(json.dumps({"run_id": run_id, **result}, ensure_ascii=True) + "\n")
                journal.flush()
                # O(N) journal writes and bounded compact summaries. Never rewrite
                # an ever-growing O(N) list after every episode (O(N^2) I/O).
                if time.monotonic() - last_summary >= 30:
                    os.fsync(journal.fileno())
                    _atomic_json(report_path, {**common, "complete": False,
                                               "completed_episodes": len(results), "failed_episodes": len(errors)})
                    last_summary = time.monotonic()
            os.fsync(journal.fileno())
        _atomic_json(report_path, {**common, "complete": not errors and len(results) == len(selected),
                                   "completed_episodes": len(results), "failed_episodes": len(errors),
                                   "results": results, "errors": errors})
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
