"""Stream selected current rows once, then fetch/align source episodes on demand.

Stage 1: ``python -m starVLA.pri4r.source_prepare --scope scope.json
--dataset-root /.../robocasa365 --output-data /.../new_run/data
--manifest /.../new_run/data/input.json [--sample]``.

Stage 2 is invoked by each GT worker for entries with ``prepare_source=true``.
It downloads public pinned-revision files with bounded retries, checks exact
source/current correspondence, and returns the mandatory alignment receipt.
Only numpy/pyarrow and Python's standard library are needed; no model or GPU.
"""
from pathlib import Path
import argparse
from bisect import bisect_right
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np

from .reader import digest_file, episode_key

REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"
REVISION = "522e4ffa3bb9d9729f79b241c65fc9e589f2659c"
COLUMNS = ("timestamp", "frame_index", "action", "observation.state")
CURRENT_COLUMNS = (*COLUMNS, "episode_index", "index")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def numeric_hash(array, *, integer=False):
    """Canonical little-endian numeric hash, including shape to avoid ambiguity."""
    dtype = np.dtype("<i8" if integer else "<f8")
    value = np.ascontiguousarray(array, dtype=dtype)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(value.shape), "dtype": dtype.str}, sort_keys=True).encode())
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def arrow_numpy(array):
    """Flatten fixed/list numeric columns without Python lists per row."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        raise ValueError("null values in training/source columns")
    if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        flat = pc.list_flatten(array)
        if flat.null_count:
            raise ValueError("null list values in training/source columns")
        lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
        if len(lengths) and not np.all(lengths == lengths[0]):
            raise ValueError("ragged model state/action column")
        width = int(lengths[0]) if len(lengths) else 0
        return flat.to_numpy(zero_copy_only=False).reshape(len(array), width)
    return array.to_numpy(zero_copy_only=False)


def select_sample_entries(entries):
    wanted = ("AdjustToasterOvenTemperature", "OpenDrawer", "AddIceCubes", "NavigateKitchen")
    selected = []
    for task in wanted:
        candidates = [e for e in entries if e.get("task_name") == task or task in str(e["source_prefix"]).split("/")]
        if not candidates:
            raise ValueError(f"required acceptance task absent from training scope: {task}")
        zero = [e for e in candidates if int(e["source_episode_index"]) == 0]
        selected.append(min(zero or candidates, key=lambda e: int(e["source_episode_index"])))
    return selected


def _cache_path(output, entry):
    return Path(output) / "current" / str(entry["dataset_name"]) / f"episode_{int(entry['episode_index']):06d}"


def _base_manifest_entry(entry, cache_dir, meta, output, scope_sha, audited_root=None, audited_receipt=None):
    result = {"source_prefix": entry["source_prefix"], "source_episode_index": int(entry["source_episode_index"]),
              "current_dataset_name": entry["dataset_name"], "current_episode_index": int(entry["episode_index"]),
              "length": int(entry["length"]), "fps": 20.0, "timestamp_source_dtype": meta["timestamp_source_dtype"],
              "current_cache_path": str(cache_dir / "columns.npz"), "current_cache_metadata": str(cache_dir / "current.json"),
              "timestamps_path": str(cache_dir / "timestamps.npy"), "source_cache_root": str(Path(output) / "source"),
              "scope_sha256": scope_sha, "prepare_source": True}
    if audited_root:
        result["audited_samples_root"] = str(audited_root)
    if audited_receipt:
        result["audited_samples_receipt"] = str(audited_receipt)
    return result


def _load_existing_current(path, entry, scope_sha):
    if not path.exists():
        return None
    meta = json.loads((path / "current.json").read_text(encoding="utf-8"))
    if meta.get("schema") != "pri4r_current_columns_v1" or meta.get("complete") is not True:
        raise ValueError(f"invalid completed current-row cache: {path}")
    expected = {"scope_sha256": scope_sha, "dataset_name": entry["dataset_name"],
                "episode_index": int(entry["episode_index"]), "source_prefix": entry["source_prefix"],
                "source_episode_index": int(entry["source_episode_index"]), "length": int(entry["length"])}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"current cache identity mismatch {key}: {path}")
    for name, sha in meta["files_sha256"].items():
        if digest_file(path / name) != sha:
            raise ValueError(f"current cache bytes changed: {path / name}")
    return meta


def _publish_current(path, entry, arrays, files, scope_sha):
    length = int(entry["length"])
    if any(len(a) != length for a in arrays.values()):
        raise ValueError("streamed current episode length mismatch")
    if not np.array_equal(arrays["frame_index"], np.arange(length)):
        raise ValueError("current frame_index does not cover exact episode order")
    if not np.array_equal(arrays["index"], np.arange(int(entry["dataset_from_index"]), int(entry["dataset_to_index"]))):
        raise ValueError("current global row indices differ from episode metadata")
    if not np.all(arrays["episode_index"] == int(entry["episode_index"])):
        raise ValueError("current cache mixes episodes")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite current training data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".incomplete-" + uuid.uuid4().hex)
    temp.mkdir()
    # Uncompressed NPZ avoids wasting CPU recompressing tiny per-episode columns;
    # only four numeric columns plus indices are staged, not videos or images.
    np.savez(temp / "columns.npz", **arrays)
    np.save(temp / "timestamps.npy", arrays["timestamp"])
    meta = {"schema": "pri4r_current_columns_v1", "complete": True, "scope_sha256": scope_sha,
            "dataset_name": entry["dataset_name"], "episode_index": int(entry["episode_index"]),
            "source_prefix": entry["source_prefix"], "source_episode_index": int(entry["source_episode_index"]),
            "length": length, "dataset_from_index": int(entry["dataset_from_index"]),
            "dataset_to_index": int(entry["dataset_to_index"]), "current_parquet_files": sorted(set(files)),
            "timestamp_source_dtype": str(arrays["timestamp"].dtype),
            "column_hashes": {c: numeric_hash(arrays[c], integer=c == "frame_index") for c in COLUMNS},
            "files_sha256": {p.name: digest_file(p) for p in temp.glob("*.np*")}}
    atomic_json(temp / "current.json", meta)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed current cache: {path}")
    os.replace(temp, path)
    return meta


def _row_groups_for_ranges(parquet, ranges):
    """Use index statistics for sample mode; missing stats conservatively scan."""
    names = parquet.schema.names
    if "index" not in names:
        raise ValueError("current parquet lacks global index")
    column = names.index("index")
    ordered = sorted(ranges)
    starts = [int(pair[0]) for pair in ordered]
    # Scope episodes are disjoint, but prefix-max ends also makes overlapping
    # ranges safe. One binary search replaces scanning all 30k intervals per RG.
    ends = np.maximum.accumulate([int(pair[1]) for pair in ordered]) if ordered else []
    selected = []
    for i in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(i).column(column).statistics
        if stats is None or not stats.has_min_max:
            selected.append(i)
        else:
            candidate = bisect_right(starts, stats.max) - 1
            if candidate >= 0 and ends[candidate] > stats.min:
                selected.append(i)
    return selected


def prepare_current(scope_path, dataset_root, output_data, manifest_path, *, sample=False,
                    audited_samples_root=None, audited_samples_receipt=None, batch_size=65536):
    import pyarrow.parquet as pq
    from .generate import episode_lock
    scope_path, dataset_root, output_data = Path(scope_path), Path(dataset_root), Path(output_data)
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope_sha = digest_file(scope_path)
    entries = scope["selected_episodes"]
    entries = select_sample_entries(entries) if sample else entries
    if len({episode_key(e["source_prefix"], e["source_episode_index"]) for e in entries}) != len(entries):
        raise ValueError("scope contains duplicate physical source episodes")
    output_data.mkdir(parents=True, exist_ok=True)
    prepared = {}
    with episode_lock(output_data / ".locks" / "current_stage1.lock"):
        pending = {}
        for entry in entries:
            cache = _cache_path(output_data, entry)
            meta = _load_existing_current(cache, entry, scope_sha)
            identity = (entry["dataset_name"], int(entry["episode_index"]))
            if meta:
                prepared[identity] = _base_manifest_entry(entry, cache, meta, output_data, scope_sha,
                                                          audited_samples_root, audited_samples_receipt)
            else:
                pending.setdefault(entry["dataset_name"], {})[int(entry["episode_index"])] = entry
        for dataset, wanted in pending.items():
            files = sorted((dataset_root / dataset / "data").glob("chunk-*/file-*.parquet"))
            if not files:
                raise FileNotFoundError(f"no current parquet files in {dataset}")
            # For sample mode begin at the earliest actual start file, and stop
            # when selected episodes have ended, including any cross-file tail.
            first = min((int(e["data/chunk_index"]), int(e["data/file_index"])) for e in wanted.values())
            collected, used_files, offsets = {}, {}, {}
            selected_ids = np.fromiter(wanted, dtype=np.int64)
            done = set()
            for file in files:
                pair = (int(file.parent.name.split("-")[-1]), int(file.stem.split("-")[-1]))
                if sample and pair < first:
                    continue
                if len(done) == len(wanted):
                    break
                parquet = pq.ParquetFile(file)
                ranges = [(int(e["dataset_from_index"]), int(e["dataset_to_index"])) for k, e in wanted.items() if k not in done]
                groups = _row_groups_for_ranges(parquet, ranges)
                if not groups:
                    continue
                for batch in parquet.iter_batches(batch_size=batch_size, row_groups=groups, columns=list(CURRENT_COLUMNS)):
                    ids = arrow_numpy(batch.column(batch.schema.get_field_index("episode_index")))
                    mask = np.isin(ids, selected_ids)
                    if not mask.any():
                        continue
                    # Convert this batch once, then split selected contiguous runs.
                    arrays = {c: arrow_numpy(batch.column(batch.schema.get_field_index(c))) for c in CURRENT_COLUMNS}
                    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
                    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                        eid = int(ids[lo])
                        if eid not in wanted:
                            continue
                        if eid in done:
                            raise ValueError("current episode appears twice or is not contiguous")
                        entry = wanted[eid]
                        expected = int(entry["length"])
                        pos = offsets.get(eid, 0)
                        size = hi - lo
                        if pos + size > expected:
                            raise ValueError("current episode exceeds metadata length")
                        if not np.array_equal(arrays["frame_index"][lo:hi], np.arange(pos, pos + size)):
                            raise ValueError("current episode frame ordering/cross-file continuity mismatch")
                        if eid not in collected:
                            collected[eid] = {c: np.empty((expected, *arrays[c].shape[1:]), dtype=arrays[c].dtype) for c in CURRENT_COLUMNS}
                            used_files[eid] = []
                        for c in CURRENT_COLUMNS:
                            collected[eid][c][pos:pos + size] = arrays[c][lo:hi]
                        used_files[eid].append(str(file))
                        offsets[eid] = pos + size
                        if pos + size == expected:
                            cache = _cache_path(output_data, entry)
                            meta = _publish_current(cache, entry, collected.pop(eid), used_files.pop(eid), scope_sha)
                            prepared[(dataset, eid)] = _base_manifest_entry(entry, cache, meta, output_data, scope_sha,
                                                                           audited_samples_root, audited_samples_receipt)
                            done.add(eid)
                            offsets.pop(eid)
                            if len(prepared) % 100 == 0:
                                print(json.dumps({"stage": "current_rows", "completed": len(prepared), "expected": len(entries)}), flush=True)
            if len(done) != len(wanted):
                raise ValueError(f"missing/incomplete selected current episodes: {sorted(set(wanted) - done)[:20]}")
        manifest = {"schema": "pri4r_generation_input_v1", "repo": REPO, "revision": REVISION,
                    "scope_sha256": scope_sha, "scope_path": str(scope_path), "sample": bool(sample),
                    "episodes": [prepared[(e["dataset_name"], int(e["episode_index"]))] for e in entries]}
        atomic_json(manifest_path, manifest)
    return manifest


def _audited_candidate(relative, entry, repo, revision):
    root, receipt_path = entry.get("audited_samples_root"), entry.get("audited_samples_receipt")
    if not root or not receipt_path:
        return None
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("repo") != repo or receipt.get("revision") != revision:
        return None
    records = {r["path"]: r for r in receipt.get("files", [])}
    record = records.get(relative)
    candidate = Path(root) / relative
    if record and candidate.is_file() and digest_file(candidate) == record["sha256"]:
        return candidate, record["sha256"]
    return None


def _retrieve(relative, entry, repo, revision, *, attempts=4, timeout=90):
    destination = Path(entry["source_cache_root"]) / relative
    receipt_path = destination.with_name(destination.name + ".download.json")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(relative, safe='/')}?download=true"
    if entry.get("source_transfer_mode") == "wait_for_relay":
        # The GT worker may hold its episode lock here. The sender deliberately
        # uses a different publication lock and atomically publishes receipt last.
        status_path = Path(entry["source_relay_status_path"])
        status_path.resolve().relative_to(Path(entry["source_cache_root"]).resolve())
        deadline = time.monotonic() + float(entry.get("source_wait_timeout_seconds", 86400))
        while not (destination.exists() and receipt_path.exists()):
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") == "failed":
                    raise RuntimeError(f"source relay failed: {status.get('error', 'see relay status')}")
                expected = entry.get("source_relay_expected_episodes")
                if status.get("status") == "complete" and (expected is None or status.get("expected_episodes") == expected):
                    raise FileNotFoundError(f"relay marked complete but source file is absent: {relative}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"bounded wait for source relay expired: {relative}")
            time.sleep(min(float(entry.get("source_wait_poll_seconds", 15)), max(.01, deadline - time.monotonic())))
    if destination.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("repo") != repo or receipt.get("revision") != revision or receipt.get("path") != relative:
            raise ValueError(f"download cache identity mismatch: {destination}")
        if digest_file(destination) != receipt["sha256"]:
            raise ValueError(f"download cache corruption: {destination}")
        return destination, receipt["sha256"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download-" + uuid.uuid4().hex)
    audited = _audited_candidate(relative, entry, repo, revision)
    if audited:
        shutil.copyfile(audited[0], temporary)
        sha = digest_file(temporary)
        if sha != audited[1]:
            raise ValueError("audited source copy changed unexpectedly")
        origin = "pinned_audited_copy"
    else:
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "StarVLA-Pri4R-GT/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                    expected_length = response.headers.get("Content-Length")
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if expected_length and temporary.stat().st_size != int(expected_length):
                    raise IOError("source download content length mismatch")
                sha = digest_file(temporary)
                break
            except (OSError, urllib.error.URLError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(2 ** attempt, 8))
        origin = "pinned_huggingface_download"
    if destination.exists():
        if digest_file(destination) != sha:
            raise ValueError(f"refusing to overwrite conflicting source bytes: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    atomic_json(receipt_path, {"repo": repo, "revision": revision, "path": relative,
                              "sha256": sha, "bytes": destination.stat().st_size, "origin": origin})
    return destination, sha


def prepare_episode(entry, repo=REPO, revision=REVISION):
    """Fetch one pinned source and verify exact alignment, under a source lock."""
    import pyarrow.parquet as pq
    from .generate import episode_lock
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be a pinned commit")
    key = episode_key(entry["source_prefix"], entry["source_episode_index"])
    cache_path = Path(entry["current_cache_path"])
    current_meta = json.loads(Path(entry["current_cache_metadata"]).read_text(encoding="utf-8"))
    expected = {"scope_sha256": entry["scope_sha256"], "dataset_name": entry["current_dataset_name"],
                "episode_index": int(entry["current_episode_index"]), "source_prefix": entry["source_prefix"],
                "source_episode_index": int(entry["source_episode_index"]), "length": int(entry["length"])}
    if current_meta.get("schema") != "pri4r_current_columns_v1" or current_meta.get("complete") is not True:
        raise ValueError("current-row cache is not complete")
    for name, value in expected.items():
        if current_meta.get(name) != value:
            raise ValueError(f"staged current identity mismatch: {name}")
    for path in (cache_path, Path(entry["timestamps_path"])):
        if digest_file(path) != current_meta["files_sha256"][path.name]:
            raise ValueError("staged current bytes differ from preparation receipt")
    with np.load(cache_path, allow_pickle=False) as current_archive:
        current = {c: current_archive[c] for c in CURRENT_COLUMNS}
    index = int(entry["source_episode_index"])
    prefix = entry["source_prefix"].strip("/")
    extra = f"{prefix}/lerobot/extras/episode_{index:06d}"
    parquet_relative = f"{prefix}/lerobot/data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
    lock = Path(entry["source_cache_root"]) / ".locks" / (hashlib.sha256(key.encode()).hexdigest() + ".lock")
    with episode_lock(lock):
        files, hashes = {}, {}
        for name in ("states.npz", "model.xml.gz", "ep_meta.json"):
            files[name], hashes[name] = _retrieve(extra + "/" + name, entry, repo, revision)
        source_parquet, parquet_sha = _retrieve(parquet_relative, entry, repo, revision)
        table = pq.read_table(source_parquet, columns=list(COLUMNS))
        source = {c: arrow_numpy(table[c]) for c in COLUMNS}
        if table.num_rows != int(entry["length"]) or not np.array_equal(source["frame_index"], np.arange(table.num_rows)):
            raise ValueError("source episode length/frame ordering mismatch")
        pairs = {}
        for column in COLUMNS:
            integer = column == "frame_index"
            a, b = np.asarray(source[column]), np.asarray(current[column])
            if a.shape != b.shape or not np.isfinite(a).all() or not np.array_equal(a, b):
                raise ValueError(f"exact source/current mismatch: {column}")
            pairs[column] = {"source": numeric_hash(a, integer=integer), "current": numeric_hash(b, integer=integer)}
            if pairs[column]["current"] != current_meta["column_hashes"][column]:
                raise ValueError(f"staged column digest changed: {column}")
        proof = {"schema": "pri4r_source_alignment_v1", "passed": True,
                 "source_prefix": prefix, "source_episode_index": index,
                 "current_episode_index": int(entry["current_episode_index"]),
                 "current_dataset_name": entry["current_dataset_name"], "length": int(entry["length"]),
                 "scope_sha256": entry["scope_sha256"], "column_hashes": pairs,
                 "source_parquet_sha256": parquet_sha, "source_files": hashes,
                 "current_cache_sha256": digest_file(cache_path)}
        proof_path = files["states.npz"].parent / "alignment.json"
        if proof_path.exists():
            previous = json.loads(proof_path.read_text(encoding="utf-8"))
            if previous != proof:
                raise ValueError("existing completed alignment receipt differs")
        else:
            atomic_json(proof_path, proof)
    result = dict(entry)
    result.update(episode_dir=str(files["states.npz"].parent), source_sha256=hashes,
                  alignment_evidence=proof, prepare_source=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--audited-samples-root")
    parser.add_argument("--audited-receipt")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    result = prepare_current(args.scope, args.dataset_root, args.output_data, args.manifest,
                             sample=args.sample, audited_samples_root=args.audited_samples_root,
                             audited_samples_receipt=args.audited_receipt, batch_size=args.batch_size)
    print(json.dumps({"manifest": args.manifest, "episodes": len(result["episodes"]),
                      "scope_sha256": result["scope_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
