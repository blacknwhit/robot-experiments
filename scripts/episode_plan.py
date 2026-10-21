"""Portable, JSON-only snapshots of real serial RoboCasa episode initializations."""
import copy,hashlib,json,os,pickle,random
from pathlib import Path
import numpy as np

FORMAT='robocasa_frozen_episode_plan_v1'
SET_FORMAT='robocasa_frozen_episode_plan_set_v1'

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(path)

def rng_snapshot(env):
    numpy=np.random.get_state();python=random.getstate()
    return {'env':copy.deepcopy(env.rng.bit_generator.state),'numpy':[numpy[0],numpy[1].tolist(),*numpy[2:]],
            'python':[python[0],list(python[1]),python[2]]}

def restore_rng(env,state):
    if state['env']['bit_generator']!=env.rng.bit_generator.state['bit_generator']:raise ValueError('RNG generator changed')
    env.rng.bit_generator.state=copy.deepcopy(state['env'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    p=state['python'];random.setstate((p[0],tuple(p[1]),p[2]))

def observation_hash(obs):
    from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import _hash_value
    h=hashlib.sha256();_hash_value(h,obs);return h.hexdigest()

def mujoco_state_hash(env):
    return hashlib.sha256(pickle.dumps(np.asarray(env.sim.get_state().flatten()),protocol=4)).hexdigest()

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_plan(path,task=None,trials=None):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format')!=FORMAT or data.get('complete') is not True:raise ValueError('Incomplete/unknown episode plan')
    n=data['num_trials'];rows=data['resets']
    if type(n) is not int or n<1 or len(rows)!=n+1 or [r['episode'] for r in rows]!=list(range(n+1)):raise ValueError('Missing/duplicate plan episodes')
    if task is not None and data['env']!=task:raise ValueError('Episode plan belongs to another task')
    if trials is not None and n<trials:raise ValueError('Episode plan has too few trials')
    if data['seed']!=7 or data['n_action_steps']!=5 or data['env_split']!='pretrain':raise ValueError('Episode protocol changed')
    for row in rows:
        if any(len(row[k])!=64 for k in ('initial_observation_sha256','initial_state_sha256')):raise ValueError('Invalid initial-state hash')
        if set(row['pre_reset_rng'])!={'env','numpy','python'}:raise ValueError('Missing RNG state')
    ref=data['reference_result']
    if ref['num_trials']!=n or len(ref['successes'])!=n or len(ref['episode_trace_sha256'])!=n:raise ValueError('Invalid reference result')
    if any(type(x) is not bool for x in ref['successes']) or sum(ref['successes'])!=ref['num_successes']:raise ValueError('Invalid reference successes')
    if len(set(ref['episode_trace_sha256']))!=n or any(len(x)!=64 for x in ref['episode_trace_sha256']):raise ValueError('Invalid reference traces')
    for k in ('env','seed','env_split','max_episode_steps','n_action_steps'):
        if ref.get(k)!=data[k]:raise ValueError('Reference protocol mismatch: '+k)
    return data

def build_plan_set(directory):
    directory=Path(directory);plans={}
    for path in sorted(directory.glob('*.json')):
        if path.name=='PLAN_SET.json' or path.name.endswith('.inprogress.json'):continue
        plan=load_plan(path)
        task=plan['env'].removeprefix('robocasa/')
        if path.name!=task+'.json':raise ValueError('Plan filename must match its task')
        plans[task]=file_sha(path)
    if not plans:raise ValueError('No complete episode plans')
    data={'format':SET_FORMAT,'plans':plans};target=directory/'PLAN_SET.json'
    if target.exists():
        if json.loads(target.read_text())!=data:raise ValueError('Plan set is immutable; choose a new directory')
    else:atomic_json(target,data)
    return file_sha(target)

def load_plan_set(directory):
    directory=Path(directory);target=directory/'PLAN_SET.json';raw=target.read_bytes();data=json.loads(raw)
    if data.get('format')!=SET_FORMAT or not data.get('plans'):raise ValueError('Invalid plan set')
    for task,digest in data['plans'].items():
        if not task or not all(c.isalnum() or c=='_' for c in task):raise ValueError('Invalid task filename')
        path=directory/(task+'.json')
        if file_sha(path)!=digest:raise ValueError('Plan set file changed: '+task)
        load_plan(path,'robocasa/'+task)
    return data['plans'],hashlib.sha256(raw).hexdigest()

class Recorder:
    def __init__(self,path):
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        self.path=Path(path);self.rows=[];self.owner=RoboCasaGymEnv;self.original=RoboCasaGymEnv.reset
        if self.path.exists():raise FileExistsError('Episode plans are immutable; choose a new output path')
        def reset(env,*args,**kwargs):
            before=rng_snapshot(env.env);obs,info=self.original(env,*args,**kwargs)
            self.rows.append({'episode':len(self.rows),'pre_reset_rng':before,
                'initial_observation_sha256':observation_hash(obs),'initial_state_sha256':mujoco_state_hash(env.env)})
            atomic_json(self.path.with_suffix('.inprogress.json'),{'format':FORMAT,'complete':False,'resets':self.rows})
            return obs,info
        self.owner.reset=reset
    def close(self):self.owner.reset=self.original
    def finalize(self,result,sim_freeze_sha256):
        import robocasa,robosuite
        n=result['num_trials']
        if result['n_envs']!=1 or len(self.rows)!=n+1:raise ValueError('Capture must use the original serial reset stream')
        data={'format':FORMAT,'complete':True,'env':result['env'],'env_split':result['env_split'],'seed':result['seed'],
            'n_action_steps':result['n_action_steps'],'max_episode_steps':result['max_episode_steps'],'num_trials':n,
            'robocasa_version':robocasa.__version__,'robosuite_version':robosuite.__version__,
            'sim_freeze_sha256':sim_freeze_sha256,'resets':self.rows,'reference_result':result}
        atomic_json(self.path,data);load_plan(self.path)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--build-set',type=Path,required=True);a=p.parse_args()
    print(build_plan_set(a.build_set))
