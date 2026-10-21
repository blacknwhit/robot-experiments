"""One task, fixed original episode IDs, asynchronous environments and batch1 RPC."""
import argparse,json,multiprocessing as mp,random,time,traceback,runpy,sys
from pathlib import Path
import numpy as np
from episode_plan import atomic_json,load_plan,restore_rng,observation_hash,mujoco_state_hash,Recorder,file_sha

def select_episode(model,episode):
    model._ensure_counters(1);model._episode_indices[0]=episode;model._replan_indices[0]=0
    model._trace_hashers[0]=model._new_trace_hasher(0)

def replay_worker(slot,ids,plan_path,plan_sha,output,host,port,barrier):
    model=env=None
    try:
        import gymnasium as gym
        import robocasa,robosuite
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        from examples.pi05_robocasa365.eval_files.simulation_env import create_single_env
        from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import PolicyWrapper
        plan=load_plan(plan_path);random.seed(plan['seed']);np.random.seed(plan['seed'])
        if file_sha(plan_path)!=plan_sha:raise ValueError('Plan changed during worker startup')
        if (robocasa.__version__,robosuite.__version__)!=(plan['robocasa_version'],plan['robosuite_version']):raise ValueError('Simulator version changed')
        reset_count=0;initial=[];original=RoboCasaGymEnv.reset
        def reset(obj,*args,**kwargs):
            nonlocal reset_count
            target=ids[reset_count] if reset_count<len(ids) else ids[-1]+1
            row=plan['resets'][target];restore_rng(obj.env,row['pre_reset_rng'])
            observation,info=original(obj,*args,**kwargs)
            obs_hash=observation_hash(observation);state_hash=mujoco_state_hash(obj.env)
            if obs_hash!=row['initial_observation_sha256'] or state_hash!=row['initial_state_sha256']:raise ValueError(f'Initial scene differs for episode {target}')
            initial.append({'episode':target,'observation_sha256':obs_hash,'state_sha256':state_hash});reset_count+=1
            return observation,info
        RoboCasaGymEnv.reset=reset
        model=PolicyWrapper(host=host,port=port,task_name=plan['env'],base_seed=plan['seed'],strict_determinism=True,trace_determinism=True)
        args=argparse.Namespace(env_name=plan['env'],env_split=plan['env_split'],seed=plan['seed'],video_out_path=None,
            n_action_steps=plan['n_action_steps'],max_episode_steps=plan['max_episode_steps'])
        env=gym.vector.SyncVectorEnv([lambda:create_single_env(args,0)])
        observations,_=env.reset();select_episode(model,ids[0])
        if barrier is not None:barrier.wait(timeout=900)
        started=time.time();current=0;success=False;requests=0;rows=[]
        while current<len(ids):
            actions=model.step(observations)['actions']
            observations,rewards,terminated,truncated,_=env.step(actions);done=np.logical_or(terminated,truncated)
            success|=bool(rewards[0]);requests+=1;model.finish_step(done)
            if done[0]:
                rows.append({'episode':ids[current],'success':success,'trace':model.episode_trace_hashes[-1],'requests':requests})
                current+=1;requests=0;success=False
                if current<len(ids):select_episode(model,ids[current])
        ended=time.time()
        atomic_json(output/f'worker{slot}.json',{'slot':slot,'ids':ids,'rows':rows,'initial_scenes':initial,'started':started,'ended':ended,'metadata':model.server_metadata})
    except BaseException:
        atomic_json(output/f'worker{slot}.error.json',{'traceback':traceback.format_exc()})
        if barrier is not None:barrier.abort()
        raise
    finally:
        try:
            if model is not None:model.close()
        finally:
            if env is not None:env.close()

def replay(a):
    plan_sha=file_sha(a.plan)
    plan=load_plan(a.plan,a.task,a.trials)
    if file_sha(a.plan)!=plan_sha:raise ValueError('Plan changed during startup')
    if not 1<=a.n_envs<=min(8,a.trials):raise ValueError('n-envs must be 1..min(8,trials)')
    a.output.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(a.n_envs) if a.barrier else None;children=[];started=time.time()
    try:
        for slot in range(a.n_envs):
            child=ctx.Process(target=replay_worker,args=(slot,list(range(slot,a.trials,a.n_envs)),a.plan,plan_sha,a.output,a.host,a.port,barrier));child.start();children.append(child)
        deadline=time.monotonic()+a.timeout
        while any(p.is_alive() for p in children):
            if any(p.exitcode not in (None,0) for p in children):raise RuntimeError('Simulation worker failed; inspect worker error evidence')
            if time.monotonic()>deadline:raise TimeoutError('Fixed-episode evaluation timeout')
            time.sleep(.2)
        if any(p.exitcode!=0 for p in children):raise RuntimeError('Simulation worker failed')
        workers=[json.loads((a.output/f'worker{i}.json').read_text()) for i in range(a.n_envs)]
        rows=sorted((r for w in workers for r in w['rows']),key=lambda r:r['episode'])
        if [r['episode'] for r in rows]!=list(range(a.trials)):raise ValueError('Duplicated/missing episode IDs')
        metadata=workers[0]['metadata']
        if any(w['metadata']!=metadata for w in workers):raise ValueError('Workers connected to different policies')
        traces=[r['trace'] for r in rows]
        if len(set(traces))!=a.trials:raise ValueError('Duplicate episode traces')
        if file_sha(a.plan)!=plan_sha:raise ValueError('Plan changed during evaluation')
        result={'complete':True,'env':plan['env'],'episode_protocol':'frozen_scene_plan_v1','episode_plan_sha256':plan_sha,
            'num_trials':a.trials,'n_envs':a.n_envs,'logical_environment_index':0,'episode_ids':list(range(a.trials)),
            'seed':plan['seed'],'n_action_steps':plan['n_action_steps'],'max_episode_steps':plan['max_episode_steps'],
            'successes':[r['success'] for r in rows],'num_successes':sum(r['success'] for r in rows),'episode_trace_sha256':traces,
            'requests':sum(r['requests'] for r in rows),'policy_server_metadata':metadata,
            'initial_scenes':sorted((r for w in workers for r in w['initial_scenes'][:-1]),key=lambda r:r['episode']),
            'rollout_timing':{'started_at':min(w['started'] for w in workers),'finished_at':max(w['ended'] for w in workers)},
            'wall_timing':{'started_at':started,'finished_at':time.time()}}
        atomic_json(a.output/'result.json',result)
    finally:
        for p in children:
            if p.is_alive():p.terminate()
            p.join(timeout=15)
            if p.is_alive():p.kill();p.join(timeout=10)
            if p.is_alive():raise RuntimeError('Could not stop an owned simulator')

def capture(a):
    rgb_report=None
    if a.rgb_last_frame:
        from rgb_last_frame import install
        rgb_report=install()
    recorder=Recorder(a.plan)
    try:
        sys.argv=['simulation_env','--host',a.host,'--port',str(a.port),'--env-name',a.task,'--env-split','pretrain','--num-trials',str(a.trials),
            '--n-envs','1','--max-episode-steps',str(a.horizon),'--n-action-steps','5','--seed','7','--checkpoint',str(a.checkpoint),
            '--profile','published_v1_0_0','--protocol-comparable','0','--horizon-multiplier','1.5','--strict-determinism','1',
            '--trace-determinism','1','--renderer-backend','egl','--result-out-path',str(a.output/'result.json')]
        runpy.run_module('examples.pi05_robocasa365.eval_files.simulation_env',run_name='__main__')
        result=json.loads((a.output/'result.json').read_text())
        if rgb_report:
            result['simulation_optimization']=rgb_report();atomic_json(a.output/'result.json',result)
        recorder.finalize(result,a.sim_freeze_sha256)
    finally:recorder.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',action='store_true');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True);p.add_argument('--trials',type=int,required=True)
    p.add_argument('--n-envs',type=int,default=1);p.add_argument('--horizon',type=int);p.add_argument('--checkpoint',type=Path);p.add_argument('--sim-freeze-sha256')
    p.add_argument('--rgb-last-frame',action='store_true')
    p.add_argument('--barrier',action='store_true',help='Validation only: synchronize initial resets, never per-step');p.add_argument('--timeout',type=int,default=14400);a=p.parse_args()
    if a.capture:capture(a)
    else:replay(a)
if __name__=='__main__':main()
