"""Resident policy per GPU, dynamic task batches, parallel CPU/single-owner RGB.

Same 50x50 published episode seeds and batch-1 model math as evaluate_seeded.py.
This execution topology has its own output identity; never mix old run folders.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import os
from pathlib import Path
from queue import Queue, Empty
import socket
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from common import ROOT, VARIANTS, activate, environment, runtime, sha256
from episode_plan import atomic_json
from evaluate import TASKS, gpu_assignments
from evaluate_task import bind_output_identity, check_port_available
from evaluate_seeded import identity_for, validate_receipt, summarize
from scene_suite import run_task_process, stop_process_group
from seeded_evaluation import PARALLEL_EXECUTION_VERSION, selected_episodes, validate_result, seed_suite


def selected_tasks(args):
    if args.all_tasks:
        return list(TASKS)
    names=args.tasks.split(',')
    if len(names)!=len(set(names)) or any(name not in {t['task'] for t in TASKS} for name in names):
        raise ValueError('Unknown or duplicate tasks')
    return [t for t in TASKS if t['task'] in names]


def parallel_identity(args):
    identity=identity_for(args)
    identity.update(simulator_execution=PARALLEL_EXECUTION_VERSION, tasks_per_gpu=args.tasks_per_gpu,
                    simulator_cpu_threads=args.simulator_cpu_threads,
                    rgb_schedule=getattr(args,'rgb_schedule','all'),
                    initial_rgb=getattr(args,'initial_rgb','all'),
                    task_scheduling='longest_horizon_first_dynamic_batches_v1')
    identity['entry_sources'].update({name:sha256(ROOT/'scripts'/name) for name in
        ('evaluate_parallel.py','parallel_seeded_sim.py','render_rpc.py','render_scene_packet.py','final_frame_rgb.py')})
    return identity


def file_signatures(args):
    """Cheap between-batch mutation guard; full hashes still checked at both ends.

    The model is resident and cannot change when a disk file changes. Avoid
    re-reading its multi-GB checkpoint after every task batch.
    """
    paths=[args.checkpoint,args.config,args.home/'receipts/sim.freeze.txt',ROOT/'provenance/source_files.sha256']
    paths.extend(ROOT/line.split('  ',1)[1] for line in (ROOT/'provenance/source_files.sha256').read_text().splitlines())
    signatures={}
    for path in paths:
        stat=path.stat()
        signatures[str(path)]=(str(path.resolve()),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    return signatures


def verify_sources():
    for line in (ROOT/'provenance/source_files.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1)
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=expected:
            raise ValueError('Source differs from sealed release: '+relative)


def committed_result(args,identity,task,ids,diagnostic):
    target=args.output/('diagnostic' if diagnostic else 'formal')/(task+'.json')
    receipt_path=target.with_suffix('.identity.json')
    if not target.exists() or not receipt_path.exists():return None
    receipt=json.loads(receipt_path.read_text())
    attempt=(args.output/receipt['evidence']).resolve()
    if not attempt.is_relative_to(args.output.resolve()):raise ValueError('Evidence path escapes output')
    marker=attempt/'ACCEPTED.json'
    if not marker.exists():return None
    accepted=json.loads(marker.read_text())
    if (accepted.get('identity')!=identity or accepted.get('results',{}).get(task)!=sha256(target)
            or json.loads((attempt/'EXIT.json').read_text()).get('returncode')!=0):
        raise ValueError('Invalid batch commit evidence')
    completed=json.loads((attempt/'simulation/COMPLETE.json').read_text())
    exits=json.loads((attempt/'simulation/worker_exit.json').read_text())['workers']
    if (task not in completed['tasks'] or not exits or any(row['exitcode']!=0 for row in exits)
            or (attempt/'simulation/ERROR.json').exists() or (attempt/'simulation/CLEANUP_ERROR.json').exists()):
        raise ValueError('Committed batch has incomplete or failed simulator evidence')
    raw=attempt/'simulation/results'/(task+'.json')
    if sha256(raw)!=receipt.get('raw_result_sha256'):raise ValueError('Raw task result changed')
    return validate_receipt(target,identity,task,ids,diagnostic,args.diagnostic_max_replans)


def summarize_committed(args,identity):
    verify_sources()
    for task in TASKS:
        if committed_result(args,identity,task['task'],list(range(50)),False) is None:
            raise ValueError('Task has not been committed: '+task['task'])
    summarize(args,identity)


def start_policy(args, identity, gpu, port, directory):
    check_port_available(port)
    env=environment(args.variant)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED='7', OPENPI_DATA_HOME=str(args.home/'cache/openpi'))
    command=[sys.executable,str(ROOT/'scripts/server.py'),'--variant',args.variant,'--config',str(args.config),
             '--checkpoint',str(args.checkpoint),'--port',str(port),'--inference-backend','cuda-graph-pipeline']
    with (directory/'server.log').open('w') as log:
        server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+1200
        while True:
            if server.poll() is not None:
                raise RuntimeError('Policy startup failed: '+str(directory/'server.log'))
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Policy startup timed out')
                time.sleep(2)
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        client=WebsocketClientPolicy(port=port)
        try: metadata=client.get_server_metadata()
        finally: client.close()
        for key in ('variant','checkpoint_sha256','config_sha256','inference_backend','inference_backend_version'):
            if metadata.get(key)!=identity[key]:raise ValueError('Foreign policy identity: '+key)
        atomic_json(directory/'POLICY_READY.json', {'pid':server.pid,'port':port,'gpu':gpu,'metadata':metadata})
        return server
    except BaseException:
        stop_process_group(server)
        raise


def run(args, identity, tasks):
    import fcntl
    activate(args.variant)
    ids=selected_episodes(args.diagnostic_episodes)
    diagnostic=args.diagnostic_episodes is not None
    category='diagnostic' if diagnostic else 'formal'
    verify_sources()
    signatures=file_signatures(args)
    queue=Queue()
    for task in sorted(tasks,key=lambda t:(-t['horizon'],t['task'])):
        if committed_result(args,identity,task['task'],ids,diagnostic) is None:
            queue.put(task['task'])
    if queue.empty():return
    gpus,egls=gpu_assignments(args.gpu_ids,args.egl_device_ids)
    failed=threading.Event()
    def slot(index):
        gpu,egl,port=gpus[index],egls[index],args.port_base+index
        directory=args.output/'services'/uuid.uuid4().hex
        directory.mkdir(parents=True)
        server=None
        try:
            with (args.output/('gpu-'+gpu.replace('/','_')+'.lock')).open('a+') as gpu_lock:
                fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                server=start_policy(args,identity,gpu,port,directory)
                while not failed.is_set():
                    batch=[]
                    for _ in range(args.tasks_per_gpu):
                        try:batch.append(queue.get_nowait())
                        except Empty:break
                    if not batch:break
                    with ExitStack() as locks:
                        for task in batch:
                            lock=locks.enter_context((args.output/(task+'.lock')).open('a+'))
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        attempt=args.output/'parallel-runs'/uuid.uuid4().hex
                        attempt.mkdir(parents=True)
                        for task in batch:
                            target=args.output/category/(task+'.json')
                            for source in (target,target.with_suffix('.identity.json')):
                                if source.exists():
                                    (attempt/'uncommitted_previous').mkdir(exist_ok=True)
                                    shutil.copy2(source,attempt/'uncommitted_previous'/source.name)
                        env=environment(args.variant,shim=False,overlay=False,cpu_threads=args.simulator_cpu_threads)
                        env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('LD_PRELOAD',None)
                        env.update(PYTHONPATH=os.pathsep.join((str(ROOT/'scripts'),str(runtime(args.variant)))),
                                   PYTHONHASHSEED='7',PYTHONFAULTHANDLER='1',MUJOCO_GL='egl',
                                   PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=egl)
                        command=[str(args.sim_python),str(ROOT/'scripts/parallel_seeded_sim.py'),'--tasks',','.join(batch),
                                 '--output',str(attempt/'simulation'),'--port',str(port),'--n-envs',str(args.n_envs),
                                 '--timeout',str(args.timeout)]
                        command+=['--rgb-schedule',args.rgb_schedule]
                        command+=['--initial-rgb',args.initial_rgb]
                        if diagnostic:command+=['--diagnostic-episodes',args.diagnostic_episodes]
                        if args.diagnostic_max_replans is not None:command+=['--diagnostic-max-replans',str(args.diagnostic_max_replans)]
                        atomic_json(attempt/'STARTED.json',{'identity':identity,'gpu':gpu,'egl':egl,'tasks':batch,
                                    'command':command,'hostname':socket.gethostname(),'started':time.time()})
                        with (attempt/'simulation.log').open('w') as log:
                            code=run_task_process(command,env=env,log=log,timeout=args.timeout+120)
                        atomic_json(attempt/'EXIT.json',{'returncode':code,'ended':time.time()})
                        if code:raise RuntimeError(f'Simulator failed {code}; see {attempt}')
                        if server.poll() is not None:raise RuntimeError('Policy exited during evaluation')
                        if file_signatures(args)!=signatures:raise ValueError('Source/weights/config changed during run')
                        checked=[]
                        for task in batch:
                            raw=attempt/'simulation/results'/(task+'.json')
                            data=json.loads(raw.read_text())
                            validate_result(data,task,ids,args.n_envs,identity,diagnostic,args.diagnostic_max_replans)
                            checked.append((task,data,sha256(raw)))
                        results={}
                        for task,data,raw_sha in checked:
                            target=args.output/category/(task+'.json')
                            atomic_json(target,data)
                            results[task]=sha256(target)
                            atomic_json(target.with_suffix('.identity.json'),{'identity':identity,'result_sha256':sha256(target),
                                        'raw_result_sha256':raw_sha,'evidence':str(attempt.relative_to(args.output))})
                        atomic_json(attempt/'ACCEPTED.json',{'tasks':batch,'diagnostic_only':diagnostic,
                                    'identity':identity,'results':results})
                        print(json.dumps({'completed_tasks':batch,'gpu':gpu,'remaining_tasks':queue.qsize()}),flush=True)
        except BaseException:
            failed.set()
            atomic_json(directory/'ERROR.json',{'traceback':traceback.format_exc()})
            raise
        finally:
            if server is not None:stop_process_group(server)
    with ThreadPoolExecutor(max_workers=min(len(gpus),queue.qsize())) as pool:
        futures=[pool.submit(slot,i) for i in range(min(len(gpus),queue.qsize()))]
        for future in as_completed(futures):future.result()
    if file_signatures(args)!=signatures or parallel_identity(args)!=identity:
        raise ValueError('Final source/weight/config integrity verification failed')
    verify_sources()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    for key in ('home','config','checkpoint','sim-python','output'):p.add_argument('--'+key,type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all-tasks',action='store_true')
    mode.add_argument('--tasks')
    mode.add_argument('--summarize',action='store_true')
    p.add_argument('--gpu-ids',default='0');p.add_argument('--egl-device-ids')
    p.add_argument('--port-base',type=int,default=19090)
    p.add_argument('--n-envs',type=int,choices=range(1,51),default=8,help='Total CPU environment workers per GPU')
    p.add_argument('--tasks-per-gpu',type=int,choices=range(1,51),default=2)
    p.add_argument('--simulator-cpu-threads',type=int,choices=range(1,9),default=1)
    p.add_argument('--rgb-schedule',choices=('all','final'),default='all')
    p.add_argument('--initial-rgb',choices=('all','deferred'),default='all')
    p.add_argument('--diagnostic-episodes');p.add_argument('--diagnostic-max-replans',type=int)
    p.add_argument('--timeout',type=int,default=28800);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    for key in ('home','config','checkpoint','sim_python','output'):setattr(args,key,getattr(args,key).resolve())
    selected_episodes(args.diagnostic_episodes)
    if args.diagnostic_max_replans is not None and (args.diagnostic_episodes is None or args.diagnostic_max_replans<1):
        p.error('Replan limit requires explicit diagnostic episodes')
    gpus,_=gpu_assignments(args.gpu_ids,args.egl_device_ids)
    if args.timeout<=0 or not 1024<=args.port_base<=65535-len(gpus):p.error('Invalid timeout/ports')
    tasks=[] if args.summarize else selected_tasks(args)
    if args.dry_run:
        print(json.dumps({'seed_suite':seed_suite(),'tasks':[t['task'] for t in tasks],
                          'episode_ids':selected_episodes(args.diagnostic_episodes),'gpus':gpus,
                          'workers_per_gpu':args.n_envs,'tasks_per_gpu':args.tasks_per_gpu},indent=2));return
    identity=parallel_identity(args)
    args.output.mkdir(parents=True,exist_ok=True);bind_output_identity(args.output,identity)
    import fcntl
    with (args.output/'orchestrator.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.summarize:summarize_committed(args,identity)
        else:
            run(args,identity,tasks)
            if args.all_tasks and args.diagnostic_episodes is None:summarize_committed(args,identity)


if __name__=='__main__':main()
