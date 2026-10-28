"""Parallel CPU episode workers sharing one main-thread EGL render owner."""
import argparse
from collections import deque
import faulthandler
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from episode_plan import atomic_json
from seeded_evaluation import selected_episodes, PARALLEL_EXECUTION_VERSION


def cpu_worker(pipe, args, slot):
    # All native/Python faults are attached to this exact worker, before cleanup.
    log = (args.output/f'worker{slot}.log').open('w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    faulthandler.enable(all_threads=True)
    import render_rpc
    render_rpc.install(pipe, defer_construction=getattr(args,'initial_rgb','all') == 'deferred')
    import seeded_async_sim
    current = None
    try:
        while True:
            current = pipe.recv()
            if current is None:
                return
            task, episode = current
            child = SimpleNamespace(**vars(args))
            child.task = task
            child.n_envs = 1
            child._episode_ids = [episode]
            child.output = args.output/'episodes'/task/str(episode)
            result = seeded_async_sim.run(child)
            pipe.send(('episode_done', task, episode, result))
    except BaseException:
        atomic_json(args.output/f'worker{slot}.error.json', {'work':current,'traceback':traceback.format_exc()})
        raise
    finally:
        pipe.close()


def merge_fragments(task, fragments, ids, n_envs, start, finish, diagnostic):
    fragments.sort(key=lambda f:f['episode_ids'][0])
    if [f['episode_ids'] for f in fragments] != [[i] for i in ids]:
        raise ValueError('Missing or duplicated episode fragments')
    first = fragments[0]
    invariant = ('env', 'episode_protocol', 'scene_seed_definition_sha256', 'logical_environment_index',
                 'seed', 'n_action_steps', 'max_episode_steps', 'diagnostic_max_replans',
                 'reference_plan_required', 'reference_observation_comparison', 'observation_health_version',
                 'policy_server_metadata', 'simulator_versions')
    invariant += ('rgb_schedule','initial_rgb')
    if any(any(f[k] != first[k] for k in invariant) for f in fragments):
        raise ValueError('Mixed fragment identity or evaluation protocol')
    data = dict(first)
    rows = [f['episodes'][0] for f in fragments]
    complete = all(row['episode_complete'] for row in rows)
    successes = [row['success'] for row in rows] if complete else None
    data.update(simulator_execution=PARALLEL_EXECUTION_VERSION, n_envs=n_envs,
                episode_ids=ids, scene_seeds=[row['scene_seed'] for row in rows], num_trials=len(ids),
                episodes=rows, complete=complete, successes=successes,
                num_successes=sum(successes) if complete else None,
                requests=sum(row['requests'] for row in rows),
                evaluation_kind='diagnostic' if diagnostic else 'formal',
                wall_timing={'started_at':start,'finished_at':finish},
                rollout_timing={'started_at':min(row['started_at'] for row in rows),
                                'finished_at':max(row['finished_at'] for row in rows)})
    data['rgb_counts'] = [f.get('rgb_counts') for f in fragments]
    return data


def run(args):
    from render_rpc import RenderOwner
    ids = selected_episodes(args.diagnostic_episodes)
    tasks = args.tasks.split(',')
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError('Task names must be unique')
    # Interleave tasks, always retain global episode IDs (never worker seeds).
    work = deque((task, episode) for episode in ids for task in tasks)
    count = min(args.n_envs, len(work))
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    atomic_json(args.output/'STARTED.json', {'pid':os.getpid(),'hostname':socket.gethostname(),
                'started':began,'tasks':tasks,'episode_ids':ids,'worker_count':count,
                'execution':PARALLEL_EXECUTION_VERSION})
    owner = RenderOwner()
    context = mp.get_context('spawn')
    workers = []
    pipes = {}
    assigned = {}
    fragments = {task:[] for task in tasks}
    completed = set()
    primary = None
    deadline = time.monotonic()+args.timeout
    render_time = 0.
    try:
        for slot in range(count):
            parent, child = context.Pipe()
            proc = context.Process(target=cpu_worker, args=(child,args,slot))
            proc.start(); child.close()
            workers.append(proc); pipes[parent]=slot
            item = work.popleft(); assigned[slot]=item; parent.send(item)
        while pipes:
            if time.monotonic() > deadline:
                raise TimeoutError('Parallel simulator exceeded timeout')
            for pipe in wait(list(pipes), timeout=.25):
                slot = pipes[pipe]
                try:
                    message = pipe.recv()
                except EOFError as error:
                    workers[slot].join(.1)
                    raise RuntimeError(f'Worker {slot} exited {workers[slot].exitcode}; work={assigned[slot]}') from error
                if message[0] == 'episode_done':
                    _, task, episode, data = message
                    if (task,episode) != assigned[slot] or (task,episode) in completed:
                        raise ValueError('Foreign/duplicate worker result')
                    if any(key[0] == slot for key in owner.contexts):
                        raise RuntimeError('Worker finished episode without releasing all render contexts')
                    completed.add((task,episode)); fragments[task].append(data)
                    atomic_json(args.output/'PROGRESS.json', {'completed':len(completed),'total':len(ids)*len(tasks),
                                'last_task':task,'last_episode':episode,'render_frames':owner.frames,
                                'render_seconds':render_time})
                    if work:
                        item=work.popleft(); assigned[slot]=item; pipe.send(item)
                    else:
                        pipe.send(None); pipe.close(); del pipes[pipe]
                    continue
                start=time.perf_counter()
                try:
                    value=owner.request(slot,message)
                    pipe.send({'ok':True,'value':value})
                except BaseException:
                    pipe.send({'ok':False,'error':traceback.format_exc()})
                    raise
                render_time += time.perf_counter()-start
            for slot,proc in enumerate(workers):
                if proc.exitcode not in (None,0):
                    raise RuntimeError(f'Worker {slot} failed: exit={proc.exitcode}, work={assigned[slot]}')
        for proc in workers:
            proc.join(30)
            if proc.exitcode != 0:
                raise RuntimeError('Worker did not exit cleanly')
    except BaseException:
        primary = traceback.format_exc()
        atomic_json(args.output/'ERROR.json', {'traceback':primary,'assigned':assigned,
                    'workers_before_cleanup':[{'pid':p.pid,'exitcode':p.exitcode} for p in workers]})
        raise
    finally:
        for proc in workers:
            if proc.is_alive(): proc.terminate()
        for proc in workers:
            proc.join(10)
            if proc.is_alive(): proc.kill(); proc.join(10)
        for pipe in pipes: pipe.close()
        try:
            owner.close()
        except BaseException:
            atomic_json(args.output/'CLEANUP_ERROR.json', {'traceback':traceback.format_exc()})
            if primary is None:
                raise
        finally:
            atomic_json(args.output/'worker_exit.json', {'workers':[{'pid':p.pid,'exitcode':p.exitcode} for p in workers]})
    for task in tasks:
        result=merge_fragments(task, fragments[task], ids, args.n_envs, began,time.time(),args.diagnostic_episodes is not None)
        atomic_json(args.output/'results'/(task+'.json'), result)
    atomic_json(args.output/'COMPLETE.json', {'tasks':tasks,'episodes':len(completed),
                'render_frames':owner.frames,'render_seconds':render_time,'render_timing':owner.timing,
                'registered_models':owner.registered_models,'peak_contexts':owner.peak_contexts,
                'wall_seconds':time.time()-began})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--n-envs',type=int,choices=range(1,51),default=8)
    p.add_argument('--diagnostic-episodes')
    p.add_argument('--diagnostic-max-replans',type=int)
    p.add_argument('--diagnostic-trace',action='store_true')
    p.add_argument('--rgb-schedule',choices=('all','final'),default='all')
    p.add_argument('--initial-rgb',choices=('all','deferred'),default='all')
    p.add_argument('--timeout',type=int,default=14400)
    args=p.parse_args()
    if args.diagnostic_max_replans is not None and (args.diagnostic_episodes is None or args.diagnostic_max_replans<1):
        p.error('Replan limit is diagnostic only')
    if args.diagnostic_trace and args.diagnostic_episodes is None:
        p.error('Trace capture is diagnostic only')
    run(args)


if __name__=='__main__':
    main()
