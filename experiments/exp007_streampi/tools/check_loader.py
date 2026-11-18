"""Compare real StreamPI model inputs or benchmark the host reader on fixed indices."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from contract import load
from environment import setup


def legacy_batches(stream, processor, size, start, stop):
    for first in range(start, stop, size):
        yield processor.batch([stream.sample(i) for i in range(first, first + size)])


def equal_inputs(old, new):
    import jax
    import numpy as np

    left, left_tree = jax.tree_util.tree_flatten_with_path(old)
    right, right_tree = jax.tree_util.tree_flatten_with_path(new)
    if left_tree != right_tree:
        raise AssertionError("Model-input structure changed")
    for (path, a), (_, b) in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b, strict=True, err_msg=str(path))


def measure(iterator, count):
    times = []
    with closing(iterator):
        for _ in range(count):
            start = perf_counter()
            batch = next(iterator)
            del batch
            times.append(perf_counter() - start)
    return {"seconds_per_batch": times, "first_s": times[0], "steady_mean_s": statistics.mean(times[1:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "bench"))
    parser.add_argument("--config", required=True, help="Resolved EXP-007 config from prepare.py configure")
    parser.add_argument("--start", type=int, default=0, help="First global sample index")
    parser.add_argument("--batches", type=int, help="Default: 1 for compare, 4 for bench")
    parser.add_argument("--workers", nargs="+", help="Ordered worker counts. In bench, 'legacy' selects the old reader")
    args = parser.parse_args()
    cfg = load(args.config)
    size = cfg["global_batch_size"]
    count = args.batches if args.batches is not None else (1 if args.mode == "compare" else 4)
    if args.start < 0 or count < (1 if args.mode == "compare" else 2):
        parser.error("Use a nonnegative start and at least 1 comparison batch or 2 timing batches")
    workers = args.workers or ([str(cfg["reader"]["num_workers"])] if args.mode == "compare"
                               else ["legacy", "0", str(cfg["reader"]["num_workers"]), "legacy"])
    if any(w != "legacy" and not w.isdecimal() for w in workers):
        parser.error("Workers must be nonnegative integers or 'legacy'")
    if args.mode == "compare" and (len(workers) != 1 or workers[0] == "legacy"):
        parser.error("compare takes one nonnegative process count")
    setup(cfg, cpu=True)
    from data import DemoStream, Processor
    from input_pipeline import Windows, batches

    stream = DemoStream(cfg)
    processor = Processor(cfg)
    windows = Windows(stream)
    stop = args.start + count * size
    result = {"mode": args.mode, "global_batch": size, "first_sample": args.start,
              "next_sample": stop, "batches": count}
    if args.mode == "compare":
        reference = legacy_batches(stream, processor, size, args.start, stop)
        candidate = batches(windows, processor, cfg, args.start, stop, workers=int(workers[0]))
        with closing(reference), closing(candidate):
            for old, new in zip(reference, candidate, strict=True):
                equal_inputs(old, new)
                del old, new
        result.update(exact_all_model_inputs=True, workers=int(workers[0]))
    else:
        result["methods"] = []
        for worker in workers:
            it = (legacy_batches(stream, processor, size, args.start, stop) if worker == "legacy"
                  else batches(windows, processor, cfg, args.start, stop, workers=int(worker)))
            row = {"reader": worker, **measure(it, count)}
            result["methods"].append(row)
            print("READER_TIMING", json.dumps(row), flush=True)
    print("LOADER_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
