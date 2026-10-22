"""Optional read-only native-debug snapshot, separate from the training owner."""
from pathlib import Path
import concurrent.futures,datetime,json,subprocess

def collect(state,native,out,get,save):
    config=state['diagnostic'];directory=out/'native_diagnostic';directory.mkdir()
    monitor=next((m for m in native['monitors'] if m['id']==config['native_monitor_id']),None)
    job=monitor['currentJobName'] if monitor else config['current_job']
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        detail,workers=list(pool.map(get,[f'/api/acp/jobs/{job}/detail?workspace=share-space',
            f'/api/acp/jobs/{job}/workers?workspace=share-space']))
    save(directory/'job.json',detail);save(directory/'workers.json',workers)
    result=dict(job=job,state=detail['item']['state'],monitor=monitor,
        live_native_identity=monitor is not None,job_start=detail['item'].get('startTime'),
        workers=[dict(name=w['name'],phase=w['phase']) for w in workers['items']],
        evidence_scope='native_current_diagnostic' if monitor else 'last_known_diagnostic_job',
        evidence_dir=str(directory),attempt_matches_worker=False)
    if not workers['items'] or all(w['phase'] in ('Pending','Creating') for w in workers['items']):
        result['deferred']='No executing diagnostic worker; do not reuse prior attempt'
    else:
        logs=[]
        for w in workers['items']:
            data=get(f"/api/acp/jobs/{job}/offline-logs?workspace=share-space&workerName={w['name']}")
            logs.extend(x for x in data.get('items',[]) if x.get('attributes',{}).get('lepton.workload-name')==job)
        save(directory/'strict_logs.json',logs)
        code='''from pathlib import Path
import json,time
r=Path(ROOT);r.resolve().relative_to(Path('/data/xiangyuyue/ULLM-zf/RobotVla'))
c=r/'control';d={'read_unix':time.time()}
for name in ('current_attempt.json','phase.json','TERMINAL.json','replay_budget.json'):
 p=c/name
 if p.exists():d[name]=json.loads(p.read_text())
phase=d.get('phase.json',{});output=phase.get('output')
if not output:
 records=d.get('replay_budget.json',{}).get('reserved',[])
 record=next((x for x in reversed(records) if x.get('attempt')==d.get('current_attempt.json',{}).get('attempt')),None)
 if record:output=record.get('output')
if output:
 p=Path(output);p.resolve().relative_to(r);d['log_tails']={}
 for name in ('driver.log','runtime_audit/native/gdb.log'):
  f=p/name
  if f.exists():
   with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-12000));d['log_tails'][name]=stream.read().decode(errors='replace')
 f=p/'runtime_audit/native/native.json'
 if f.exists():
  proof=json.loads(f.read_text());d['native_capture']={k:proof.get(k) for k in ('signal','exit_code','timeout_triggered','thread apply all bt 40')}
print(json.dumps(d))
'''.replace('ROOT',repr(config['target_root']))
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',state.get('reader','fslgpu'),'python3 -u -'],
            input=code,capture_output=True,text=True,encoding='utf-8',timeout=35)
        if proc.returncode:raise RuntimeError(proc.stderr)
        app=json.loads(proc.stdout);save(directory/'shared.json',app);attempt=app.get('current_attempt.json',{})
        start=result['job_start'];start=datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp() if start else None
        matches=attempt.get('hostname') in {w['name'] for w in workers['items']} and start is not None and attempt.get('started_unix',0)>=start
        result['attempt_matches_worker']=bool(matches)
        if matches:
            result['application']=app;result['phase']=app.get('phase.json',{}).get('phase')
            result['outcome']=app.get('TERMINAL.json',{}).get('outcome')
        else:result['application_evidence_rejected']='Diagnostic attempt does not match worker/start'
    save(directory/'report.json',result);save(out.parent.parent/'latest_diagnostic.json',result)
    return {k:v for k,v in result.items() if k not in ('application','monitor')}
