"""One native SCO owner, bounded smoke -> 1500-step validation -> EGL smoke.

Every training process has its own full-state run. Engineering weights never
become a formal run. Known application failures HOLD instead of retry loops.
"""
import pyarrow  # noqa: F401
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid

BASE = Path("/data/xiangyuyue/ULLM-zf/RobotVla")
ROOT = BASE / "starVLA_pi05_embedding_position_v3_debug_20261020"
REPO = ROOT / "repo"
HOME = ROOT / "home"
CONTROL = ROOT / "control"
VARIANTS = ("embedding3d_pos_a", "embedding3d_temporal_b")
sys.path.insert(0, str(REPO / "scripts"))
from common import activate, environment, sha256
activate(VARIANTS[0])
from checkpoint import choose_latest, validate
import yaml


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    assert Path(__file__).resolve().parent == ROOT
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / "pipeline.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (CONTROL / "HOLD_FOR_MAIN.json").exists():
        print("HOLD_FOR_MAIN: inspect existing evidence; no competing retry/base restart", flush=True)
        return
    if (CONTROL / "BOUNDED_VALIDATION_COMPLETE.json").exists():
        print("BOUNDED_VALIDATION_ALREADY_COMPLETE", flush=True)
        return
    release = json.loads((ROOT / "release_manifest.json").read_text())
    for rel, digest in release["files"].items():
        p = ROOT / rel
        p.resolve().relative_to(ROOT)
        assert sha256(p) == digest, rel
    tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt_dir = CONTROL / "attempts" / tag
    attempt_dir.mkdir(parents=True)
    attempt = dict(attempt=tag, hostname=socket.gethostname(), pid=os.getpid(), started_unix=time.time(),
                   world_size=2, per_device_batch=32, global_batch=64, accumulation=1,
                   max_short_steps=1500, formal_training=False, release_sha256=sha256(ROOT / "release_manifest.json"))
    save(CONTROL / "current_attempt.json", attempt)
    save(attempt_dir / "attempt.json", attempt)

    def phase(name, **fields):
        value = dict(phase=name, time_unix=time.time(), **attempt, **fields)
        save(CONTROL / "phase.json", value)
        save(attempt_dir / "phase.json", value)
        print("POSITION_VALIDATION_PHASE " + json.dumps(value), flush=True)

    def run(command, logname, env=None):
        with (attempt_dir / logname).open("x") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in child.stdout:
                    log.write(line); log.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
        if code:
            raise RuntimeError(f"{logname} exited {code}; inspect preserved current-attempt log")

    def train(variant, stage, stop):
        cfgpath = ROOT / "configs" / f"{variant}_{stage}_{stop}.yaml"
        cfg = yaml.safe_load(cfgpath.read_text())
        output = Path(cfg["run_root_dir"]) / cfg["run_id"]
        receipt = output / "portable_recipe.json"
        resume = receipt.exists()
        selected = None
        if resume:
            selected, rejected = choose_latest(output, cfg)
            save(attempt_dir / f"{variant}_{stage}_{stop}_selection.json",
                 dict(selected=str(selected), rejected=rejected))
            if int(selected.name.split("-")[-1]) >= stop:
                return output, cfgpath
        metrics = output / "point_track_metrics.jsonl"
        phase(f"{variant}_{stage}_to_{stop}", run=str(output), resume=resume,
              checkpoint=str(selected) if selected else None,
              metric_start_offset=metrics.stat().st_size if metrics.exists() else 0)
        cmd = [sys.executable, str(REPO / "scripts/train.py"), "--config", str(cfgpath)]
        if resume:
            cmd.append("--resume")
        run(cmd, f"{variant}_{stage}_{stop}.log")
        validate(output / "checkpoints" / f"checkpoint-{stop}", cfg)
        return output, cfgpath

    try:
        for v in VARIANTS:
            run([sys.executable, str(REPO / "scripts/import_preflight.py"), "--config",
                 str(ROOT / "configs" / f"{v}_smoke_2.yaml")], f"{v}_imports.log", environment(v))
        phase("cuda_environment_preflight")
        run([sys.executable, str(ROOT / "cuda_environment_probe.py"), "--output",
             str(attempt_dir / "cuda_environment.json")], "cuda_environment.log")
        import torch
        if torch.cuda.device_count() != 2 or not all("H100" in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("Expected actual 2xH100 for this validation")
        phase("hardware_nccl")
        run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
             str(REPO / "scripts/nccl_probe.py")], "nccl.log")
        for variant in VARIANTS:
            accepted = CONTROL / f"{variant}_GPU_ACCEPTED.json"
            if accepted.exists():
                assert json.loads(accepted.read_text())["release_sha256"] == attempt["release_sha256"]
                continue
            output, _ = train(variant, "smoke", 2)
            output, cfgpath = train(variant, "smoke", 3)
            rows = {r["step"]: r for r in map(json.loads, (output / "point_track_metrics.jsonl").read_text().splitlines())}
            for step in (1, 2, 3):
                assert all(math.isfinite(rows[step][k]) and rows[step][k] > 0
                           for k in ("total_loss", "action_dit_loss", "position_loss", "difference_loss", "gradient_norm"))
                assert (output / f"checkpoints/checkpoint-{step}/PORTABLE_COMPLETE.json").is_file()
            restores = []
            for rank in (0, 1):
                proof = json.loads((output / f"resume_partition_audit_rank{rank}.json").read_text())
                assert proof["completed_steps"] == 2
                assert all(proof[k] for k in ("master_and_adam_moments_bitwise_equal", "bf16_model_matches_master",
                                              "rng_restore_verified_before_child_seed", "nonparameter_padding_verified_zero"))
                restores.append(proof)
                records = {r["step"]: r for r in map(json.loads, (output / f"position_native_audit_rank{rank}.jsonl").read_text().splitlines())}
                for step in (1, 2, 3):
                    assert all(math.isfinite(x) and x > 0 for v in records[step]["masters"].values() for x in v.values())
            phase(f"{variant}_point_free_inference")
            checkpoint = output / "checkpoints/steps_3_model.safetensors"
            inference = CONTROL / f"{variant}_inference.json"
            run([sys.executable, str(REPO / "scripts/point_inference_probe.py"), "--config", str(cfgpath),
                 "--checkpoint", str(checkpoint), "--sample", str(output / "engineering_sample.pt"),
                 "--output", str(inference)], f"{variant}_inference.log", environment(variant))
            assert json.loads(inference.read_text())["passed"]
            save(accepted, dict(passed=True, engineering_only=True, run=str(output), restores=restores,
                                release_sha256=attempt["release_sha256"], inference=json.loads(inference.read_text())))
        for variant in VARIANTS:
            output, cfgpath = train(variant, "short", 1500)
            rows = {r["step"]: r for r in map(json.loads, (output / "point_track_metrics.jsonl").read_text().splitlines())}
            assert 1500 in rows and all(math.isfinite(r["total_loss"]) for r in rows.values())
            phase(f"{variant}_EGL_smoke")
            evaluation = HOME / "evaluation" / variant
            run([sys.executable, str(REPO / "scripts/evaluate.py"), "--variant", variant,
                 "--home", str(HOME), "--config", str(cfgpath), "--checkpoint",
                 str(output / "checkpoints/steps_1500_model.safetensors"),
                 "--sim-python", "/root/anaconda3/envs/robocasa365/bin/python",
                 "--output", str(evaluation), "--gpu-ids", "0", "--smoke"],
                f"{variant}_EGL.log", environment(variant))
            assert json.loads((evaluation / "SMOKE_ACCEPTED.json").read_text())["application_smoke_passed"]
        phase("bounded_validation_complete")
        save(CONTROL / "BOUNDED_VALIDATION_COMPLETE.json",
             dict(attempt, passed=True, variants=list(VARIANTS), short_steps=1500,
                  full_benchmark=False, auxiliary_coefficient=1.0, coefficient_calibrated=False))
    except Exception as exc:
        proof = dict(error=repr(exc), traceback=traceback.format_exc(), **attempt)
        save(CONTROL / "HOLD_FOR_MAIN.json", proof)
        phase("application_error_hold", error=repr(exc))
        traceback.print_exc()
        # A known application failure is held once, not blindly retrained by native auto10.


if __name__ == "__main__":
    main()
