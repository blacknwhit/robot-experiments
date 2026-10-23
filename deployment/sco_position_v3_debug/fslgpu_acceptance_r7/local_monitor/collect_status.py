"""Bounded read-only collector. Native monitor currentJobName owns identity."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import subprocess
import urllib.request

D = Path(__file__).resolve().parent


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:5177" + path, timeout=30) as f:
        value = json.load(f)
    if not value.get("ok"):
        raise RuntimeError(value)
    return value


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    state = json.loads((D / "status.json").read_text(encoding="utf-8"))
    now = datetime.datetime.now().astimezone()
    out = D / "checks" / now.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    native = get("/api/acp/auto-retrain?workspace=share-space")
    save(out / "native.json", native)
    mid = state.get("native_monitor_id")
    monitor = next((m for m in native["monitors"] if m["id"] == mid), None)
    # A completed/HOLD job is removed by native auto-retrain. Keep auditing the
    # last known job as historical evidence; never invent a live recovery owner.
    job = monitor["currentJobName"] if monitor else state.get("current_job")
    if not job:
        raise RuntimeError("No native identity or last-known job available")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        detail, workers = list(pool.map(get, [
            f"/api/acp/jobs/{job}/detail?workspace=share-space",
            f"/api/acp/jobs/{job}/workers?workspace=share-space"]))
    save(out / "job.json", detail); save(out / "workers.json", workers)
    logs = []
    for worker in workers["items"]:
        if worker["phase"] not in ("Pending", "Creating"):
            data = get(f"/api/acp/jobs/{job}/offline-logs?workspace=share-space&workerName={worker['name']}")
            logs += [x for x in data.get("items", [])
                     if x.get("attributes", {}).get("lepton.workload-name") == job]
    save(out / "strict_logs.json", logs)
    report = dict(checked_at=now.isoformat(), evidence_dir=str(out), identity_verified=monitor is not None, job=job,
                  evidence_scope="native_current_job" if monitor else "last_known_job_audit_no_native_owner",
                  state=detail["item"]["state"], job_start=detail["item"].get("startTime"),
                  monitor=monitor, workers=[dict(name=w["name"], phase=w["phase"]) for w in workers["items"]],
                  strict_log_count=len(logs), attempt_matches_current_worker=False, gpu_accepted=False)
    if state.get('diagnostic'):
        try:
            from collect_native_diagnostic import collect
            report['diagnostic']=collect(state,native,out,get,save)
        except Exception as exc:
            report['diagnostic_collection_error']=repr(exc)
    if state.get('fslgpu_acceptance'):
        try:
            from collect_fslgpu_acceptance import collect
            report['fslgpu_acceptance']=collect(state,out,get,save)
        except Exception as exc:
            report['fslgpu_acceptance_collection_error']=repr(exc)
    if not workers["items"] or all(w["phase"] in ("Pending", "Creating") for w in workers["items"]):
        report["application_deferred"] = "No executing worker; do not read or reuse a previous application attempt"
        save(out / "report.json", report); save(D / "latest_status.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "monitor"}, ensure_ascii=False))
        return
    code = '''from pathlib import Path
import json,time
r=Path(ROOT);r.resolve().relative_to(Path('/data/xiangyuyue/ULLM-zf/RobotVla'))
c=r/CONTROL_NAME;c.resolve().relative_to(r);out={'read_unix':time.time()}
for f in ['current_attempt.json','phase.json','HOLD_FOR_MAIN.json','BOUNDED_VALIDATION_COMPLETE.json',
          'embedding3d_pos_a_GPU_ACCEPTED.json','embedding3d_temporal_b_GPU_ACCEPTED.json','B_SHORT1500_COMPLETE.json']:
 p=c/f
 if p.exists():out[f]=json.loads(p.read_text())
a=out.get('current_attempt.json',{});phase=out.get('phase.json',{})
if a:
 p=c/'attempts'/a['attempt'];p.resolve().relative_to(c.resolve());out['logs']={}
 for f in p.glob('*.log'):
  with f.open('rb') as z:
   z.seek(max(0,f.stat().st_size-10000));out['logs'][f.name]=z.read().decode(errors='replace')
if phase.get('run') and 'metric_start_offset' in phase:
 p=Path(phase['run']);p.resolve().relative_to(r)
 f=p/'point_track_metrics.jsonl'
 if f.exists():
  with f.open('rb') as z:z.seek(phase['metric_start_offset']);lines=z.read().decode().splitlines()
  rows=[]
  for line in lines:
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
  out['current_stage_metrics']={'offset':phase['metric_start_offset'],'count':len(rows),'first':rows[:3],'last':rows[-5:]}
print(json.dumps(out))
'''.replace("ROOT", repr(state["target_root"])).replace('CONTROL_NAME',repr(state.get('training_control_subdir','control')))
    try:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                               state.get("reader", "fslcpu"), "python3 -u -"],
                              input=code, capture_output=True, text=True, encoding="utf-8", timeout=35)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        shared = json.loads(proc.stdout); save(out / "shared.json", shared)
        attempt = shared.get("current_attempt.json", {})
        hostname_ok = bool(attempt) and attempt.get("hostname") in {w["name"] for w in workers["items"]}
        start = report.get("job_start")
        job_start = datetime.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() if start else None
        started = attempt.get("started_unix")
        time_ok = isinstance(started, (int, float)) and job_start is not None and started >= job_start
        report["attempt_matches_current_worker"] = bool(hostname_ok and time_ok)
        report["attempt_hostname_matches"] = hostname_ok
        report["attempt_start_after_job_start"] = time_ok
        if report["attempt_matches_current_worker"]:
            report["application"] = shared
            report["gpu_accepted"] = all(shared.get(v + "_GPU_ACCEPTED.json", {}).get("passed")
                                         for v in ("embedding3d_pos_a", "embedding3d_temporal_b"))
        else:
            report["application_evidence_rejected"] = "Shared attempt does not match the native current worker"
    except Exception as exc:
        report["transport_error"] = repr(exc)
    save(out / "report.json", report); save(D / "latest_status.json", report)
    compact = {k: v for k, v in report.items() if k not in ("application", "monitor")}
    if "application" in report:
        app = report["application"]
        compact["phase"] = app.get("phase.json", {}).get("phase")
        compact["hold"] = app.get("HOLD_FOR_MAIN.json", {}).get("error")
        compact["metrics"] = app.get("current_stage_metrics")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
