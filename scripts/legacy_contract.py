"""Exact legacy content checks; no inference from matching success rates alone."""
import hashlib,json
from pathlib import Path


def require_same_policy(plan,identity):
    source=plan['reference_result'].get('policy_server_metadata',{})
    for key in ('variant','checkpoint_sha256','config_sha256'):
        if source.get(key)!=identity[key]:
            raise ValueError('Legacy replay requires this checkpoint/config own serial plan: '+key)


def require_same_trajectories(result,reference,trials):
    n=reference.get('num_trials')
    if type(n) is not int or n<1:raise ValueError('Invalid historical episode count')
    successes=reference.get('successes',[]);traces=reference.get('episode_trace_sha256',[])
    if len(successes)!=n or any(type(x) is not bool for x in successes) or sum(successes)!=reference.get('num_successes'):raise ValueError('Invalid historical successes')
    if len(traces)!=n or len(set(traces))!=n or any(len(x)!=64 for x in traces):raise ValueError('Invalid historical traces')
    if reference.get('env_split')!='pretrain' or reference.get('n_envs')!=1:raise ValueError('Reference must be the native legacy serial evaluation')
    for key in ('env','seed','n_action_steps','max_episode_steps'):
        if result.get(key)!=reference.get(key):raise ValueError('Legacy protocol mismatch: '+key)
    if reference['num_trials']<trials:raise ValueError('Historical reference has too few episodes')
    for key in ('successes','episode_trace_sha256'):
        if len(result[key])!=trials or result[key]!=reference[key][:trials]:raise ValueError('Legacy content differs: '+key)


def historical_inventory(directory,tasks):
    directory=Path(directory);files={}
    for task in tasks:
        matches=list(directory.rglob(task['task']+'.json'))
        if len(matches)>1:raise ValueError('Ambiguous historical task results: '+task['task'])
        if matches:
            path=matches[0];files[task['task']]={'path':str(path.relative_to(directory)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if not files:raise ValueError('No historical raw task results')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return files,digest


def check_historical(directory,inventory,task,result,identity,trials):
    record=inventory[task];path=Path(directory)/record['path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Historical reference changed')
    reference=json.loads(raw)
    metadata=reference.get('policy_server_metadata',{})
    # Older native results may not carry a checkpoint SHA. Exact input/action
    # traces are still required, and we never label a missing SHA as verified.
    if 'checkpoint_sha256' in metadata and metadata['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Historical policy identity mismatch: checkpoint_sha256')
    if 'variant' in metadata and metadata['variant']!=identity['variant']:
        alias=(identity['variant'],metadata['variant']) in {('query3d_v2','query'),('embedding3d_v2','embedding')}
        if not alias or metadata.get('checkpoint_sha256')!=identity['checkpoint_sha256']:raise ValueError('Historical policy identity mismatch: variant')
    if reference.get('robocasa_version','1.0.0')!='1.0.0':raise ValueError('Historical simulator version changed')
    if reference.get('n_envs')!=1:raise ValueError('Historical reference is not the legacy serial protocol')
    require_same_trajectories(result,reference,trials)
    return {'result_sha256':record['sha256'],'input_and_executed_action_traces_exact':True,
            'checkpoint_sha_verified':metadata.get('checkpoint_sha256')==identity['checkpoint_sha256']}
