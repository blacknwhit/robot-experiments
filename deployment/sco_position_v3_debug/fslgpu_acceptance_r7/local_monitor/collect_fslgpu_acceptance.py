"""Read-only CCI identity-bound acceptance; never launches or repairs work."""
import datetime,json,subprocess
def collect(state,out,get,save):
    config=state['fslgpu_acceptance'];directory=out/'fslgpu_acceptance';directory.mkdir()
    app=get('/api/cci/apps/'+config['app_id']+'/detail?workspace=share-space');save(directory/'app.json',app)
    instances=app.get('instances',{}).get('instances',[])
    result={'checked_at':datetime.datetime.now().astimezone().isoformat(),'app':config['app_id'],
            'state':app['item']['state'],'instances':[{'name':i['name'],'state':i['state']} for i in instances],
            'attempt_matches_instance':False,'accepted':False,'evidence_dir':str(directory)}
    if len(instances)!=1 or instances[0]['state']!='RUNNING':
        result['deferred']='No single live reader instance; previous application evidence is not reused'
    else:
        code='''from pathlib import Path
import json,os
r=Path(ROOT);r.resolve().relative_to(Path('/data/xiangyuyue/ULLM-zf/RobotVla'))
d={}
for name in ('current_attempt.json','phase.json','cases.json','TERMINAL.json','FSLGPU_EVALUATION_ACCEPTED.json'):
 p=r/'control'/name
 if p.exists():d[name]=json.loads(p.read_text())
p=d.get('phase.json',{}).get('output')
if p:
 p=Path(p);boundary=Path(EVIDENCE_BOUNDARY);boundary.resolve().relative_to(Path('/data/xiangyuyue/ULLM-zf/RobotVla'));p.resolve().relative_to(boundary);d['tails']={}
 for name in ('driver.log','runtime_audit/sim_0.json','runtime_audit/native/gdb.log'):
  f=p/name
  if f.exists():
   with f.open('rb') as z:z.seek(max(0,f.stat().st_size-4000));d['tails'][name]=z.read().decode(errors='replace')
print(json.dumps(d))
'''.replace('ROOT',repr(config['target_root'])).replace('EVIDENCE_BOUNDARY',repr(config.get('evidence_root',config['target_root'])))
        try:
            proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',config.get('reader','fslgpu'),'python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=30)
            if proc.returncode:raise RuntimeError(proc.stderr)
            evidence=json.loads(proc.stdout);save(directory/'shared.json',evidence)
            attempt=evidence.get('current_attempt.json',{});instance=instances[0]
            start=instance.get('last_started_time')
            epoch=datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp() if start else None
            matches=attempt.get('hostname')==instance['name'] and epoch is not None and attempt.get('started_unix',0)>=epoch
            result['attempt_matches_instance']=bool(matches)
            if matches:
                result['application']=evidence;result['phase']=evidence.get('phase.json',{}).get('phase')
                result['outcome']=evidence.get('TERMINAL.json',{}).get('outcome')
                result['accepted']=evidence.get('FSLGPU_EVALUATION_ACCEPTED.json',{}).get('passed') is True
            else:result['application_evidence_rejected']='Hostname/start do not match the current CCI instance'
        except (OSError,subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as exc:result['transport_error']=repr(exc)
    save(directory/'report.json',result);save(out.parent.parent/'latest_fslgpu_acceptance.json',result)
    return {k:v for k,v in result.items() if k!='application'}
