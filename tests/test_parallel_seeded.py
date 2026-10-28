import copy
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import subprocess
import threading
import os
from queue import Queue
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from parallel_seeded_sim import merge_fragments
from evaluate_parallel import selected_tasks
from seeded_evaluation import task_seeds, PARALLEL_EXECUTION_VERSION
from episode_plan import atomic_json
from common import sha256


class FragmentTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'),'POSIX session cleanup')
    def test_cancelled_simulator_leaves_no_running_child(self):
        import evaluate_parallel
        started=[];real_popen=subprocess.Popen
        def spawn(*args,**kwargs):
            p=real_popen(*args,**kwargs);started.append(p);return p
        cancelled=threading.Event();cancelled.set()
        with tempfile.TemporaryFile(mode='w+') as log, patch.object(evaluate_parallel.subprocess,'Popen',side_effect=spawn):
            with self.assertRaises(InterruptedError):
                evaluate_parallel.run_simulator([sys.executable,'-c','import time;time.sleep(60)'],
                    env=dict(os.environ),log=log,timeout=60,cancelled=cancelled)
        self.assertIsNotNone(started[0].poll())

    def test_large_task_batch_still_reserves_work_for_every_gpu_without_duplicates(self):
        from evaluate_parallel import reserve_initial_batches
        for gpu_count in (1,2,4,8,50):
            for requested in (1,2,8,50):
                queue=Queue()
                for task in range(50):queue.put(task)
                batches,size=reserve_initial_batches(queue,gpu_count,requested)
                self.assertEqual(len(batches),gpu_count)
                self.assertTrue(all(batches))
                self.assertLessEqual(size,requested)
                ids=[task for batch in batches for task in batch]
                while not queue.empty():ids.append(queue.get_nowait())
                self.assertEqual(sorted(ids),list(range(50)))

    def test_real_task_horizons_do_not_concentrate_long_work_on_first_gpu(self):
        from evaluate_parallel import reserve_initial_batches, TASKS
        ordered=sorted(TASKS,key=lambda task:(-task['horizon'],task['task']))
        weights={task['task']:task['horizon'] for task in ordered}
        for gpu_count in (2,4,8):
            queue=Queue()
            for task in ordered:queue.put(task['task'])
            batches,size=reserve_initial_batches(queue,gpu_count,8,weights)
            assigned=[name for batch in batches for name in batch]
            remaining=[]
            while not queue.empty():remaining.append(queue.get_nowait())
            self.assertEqual(sorted(assigned+remaining),sorted(weights))
            self.assertEqual(len(assigned),len(set(assigned)))
            self.assertTrue(all(len(batch)==size for batch in batches))
            loads=[sum(weights[name] for name in batch) for batch in batches]
            old_loads=[sum(task['horizon'] for task in ordered[i*size:(i+1)*size]) for i in range(gpu_count)]
            self.assertLess(max(loads)-min(loads),max(old_loads)-min(old_loads))
            self.assertLessEqual(max(loads)/min(loads),1.25)

    def test_one_gpu_keeps_exact_batch_order_and_pending_work(self):
        from evaluate_parallel import reserve_initial_batches, TASKS
        ordered=sorted(TASKS,key=lambda task:(-task['horizon'],task['task']))
        names=[task['task'] for task in ordered];weights={task['task']:task['horizon'] for task in ordered}
        for requested in (1,2,8,50):
            queue=Queue()
            for name in names:queue.put(name)
            batches,size=reserve_initial_batches(queue,1,requested,weights)
            self.assertEqual(batches,[names[:requested]])
            self.assertEqual(size,requested)
            pending=[]
            while not queue.empty():pending.append(queue.get_nowait())
            self.assertEqual(pending,names[requested:])

    def test_deferred_constructor_scope_restores_on_error_and_requires_hard_reset(self):
        import render_rpc
        with patch.object(render_rpc,'_defer_enabled',True), patch.object(render_rpc,'_construction_checked',True):
            self.assertFalse(render_rpc._in_construction)
            with self.assertRaisesRegex(ValueError,'constructor'):
                with render_rpc.construction_scope():
                    self.assertTrue(render_rpc._in_construction)
                    raise ValueError('constructor')
            self.assertFalse(render_rpc._in_construction)
            with self.assertRaises(RuntimeError):
                render_rpc.validate_final_reset(SimpleNamespace(unwrapped=SimpleNamespace(env=SimpleNamespace(hard_reset=False))))
            render_rpc.validate_final_reset(SimpleNamespace(unwrapped=SimpleNamespace(env=SimpleNamespace(hard_reset=True))))

    def test_resume_requires_committed_batch_and_clean_raw_worker_evidence(self):
        from evaluate_parallel import committed_result
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);attempt=output/'parallel-runs/a';target=output/'formal/CloseBlenderLid.json'
            args=SimpleNamespace(output=output,diagnostic_max_replans=None)
            identity={'test':'fixed'}
            atomic_json(target,{'result':'complete'})
            self.assertIsNone(committed_result(args,identity,'CloseBlenderLid',list(range(50)),False))
            raw=attempt/'simulation/results/CloseBlenderLid.json';atomic_json(raw,{'result':'complete'})
            atomic_json(target.with_suffix('.identity.json'),{'identity':identity,'evidence':'parallel-runs/a',
                        'result_sha256':sha256(target),'raw_result_sha256':sha256(raw)})
            self.assertIsNone(committed_result(args,identity,'CloseBlenderLid',list(range(50)),False))
            atomic_json(attempt/'EXIT.json',{'returncode':0})
            atomic_json(attempt/'simulation/COMPLETE.json',{'tasks':['CloseBlenderLid']})
            atomic_json(attempt/'simulation/worker_exit.json',{'workers':[{'pid':123,'exitcode':0}]})
            atomic_json(attempt/'ACCEPTED.json',{'identity':identity,'results':{'CloseBlenderLid':sha256(target)}})
            with patch('evaluate_parallel.validate_receipt',return_value={'accepted':True}):
                self.assertEqual(committed_result(args,identity,'CloseBlenderLid',list(range(50)),False),{'accepted':True})
                atomic_json(attempt/'simulation/worker_exit.json',{'workers':[{'pid':123,'exitcode':-9}]})
                with self.assertRaises(ValueError):committed_result(args,identity,'CloseBlenderLid',list(range(50)),False)

    def test_explicit_registry_releases_orphaned_native_context(self):
        import render_rpc
        class Context:
            _registered=True
            _remote_id=123
            _pixels=None
        context=Context()
        with patch.object(render_rpc,'_registered_contexts',{123:context}), patch.object(render_rpc,'call') as rpc:
            del context
            render_rpc.release_all()
            rpc.assert_called_once_with('release',123)
            self.assertEqual(render_rpc._registered_contexts,{})

    def test_complete_fragments_keep_global_seed_and_reject_duplicate_missing_or_mixed_identity(self):
        _,seeds=task_seeds('CloseBlenderLid')
        fragments=[]
        for i in (3,0):
            fragments.append(dict(env='robocasa/CloseBlenderLid',episode_protocol='fixed_episode_seeds_v1',
                scene_seed_definition_sha256='seeds',logical_environment_index=0,seed=7,n_action_steps=5,
                max_episode_steps=900,diagnostic_max_replans=None,reference_plan_required=False,rgb_schedule='all',initial_rgb='all',
                reference_observation_comparison=False,observation_health_version='health',
                policy_server_metadata={'checkpoint':'same'},simulator_versions={'sim':'same'},
                episode_ids=[i],episodes=[dict(episode=i,scene_seed=seeds[i],episode_complete=True,
                    success=bool(i),requests=180,started_at=2.,finished_at=10.)]))
        merged=merge_fragments('CloseBlenderLid',copy.deepcopy(fragments),[0,3],8,1.,11.,True)
        self.assertEqual(merged['episode_ids'],[0,3])
        self.assertEqual(merged['scene_seeds'],[seeds[0],seeds[3]])
        self.assertEqual(merged['requests'],360)
        self.assertEqual(merged['successes'],[False,True])
        self.assertEqual(merged['simulator_execution'],PARALLEL_EXECUTION_VERSION)
        for bad in ([fragments[0]], [fragments[0],fragments[0]]):
            with self.assertRaises(ValueError):merge_fragments('CloseBlenderLid',bad,[0,3],8,1.,11.,True)
        bad=copy.deepcopy(fragments);bad[0]['policy_server_metadata']={'checkpoint':'different'}
        with self.assertRaises(ValueError):merge_fragments('CloseBlenderLid',bad,[0,3],8,1.,11.,True)

    def test_task_queue_cannot_duplicate_or_silently_drop_unknown_tasks(self):
        for names in ('CloseBlenderLid,CloseBlenderLid','CloseBlenderLid,unknown',''):
            with self.assertRaises(ValueError):selected_tasks(SimpleNamespace(all_tasks=False,tasks=names))
        self.assertEqual(len(selected_tasks(SimpleNamespace(all_tasks=True))),50)


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Native MuJoCo required')
class SceneTransportTests(unittest.TestCase):
    def test_scene_roundtrip_preserves_native_geometry_and_rejects_capacity_mismatch(self):
        import mujoco
        from render_scene_packet import ScenePacket
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/>'
            '<geom type="plane" size="1 1 .1"/><body pos="0 0 .5"><freejoint/>'
            '<geom type="sphere" size=".1" rgba=".2 .4 .8 .5"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        scene=mujoco.MjvScene(model,maxgeom=10000)
        mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
        source=ScenePacket(scene);packet=source.capture()
        target=ScenePacket(mujoco.MjvScene(model,maxgeom=10000));target.restore(packet)
        again=target.capture()
        self.assertEqual(packet['pods'],again['pods'])
        self.assertEqual(packet['scalars'],again['scalars'])
        for key in packet['arrays']:np.testing.assert_array_equal(packet['arrays'][key],again['arrays'][key])
        bad=copy.deepcopy(packet);bad['scalars']['ngeom']=10001
        with self.assertRaises(ValueError):target.restore(bad)
        bad=copy.deepcopy(packet);bad['pods']['camera']=b''
        with self.assertRaises(ValueError):target.restore(bad)


if __name__=='__main__':unittest.main()
