import copy
import gc
import json
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_seeded
import fresh_scene
import seeded_episode_sim
import egl_lifecycle
from episode_plan import atomic_json, file_sha
from evaluate import TASKS
from scene_suite import DEFINITION
from seeded_evaluation import (PROTOCOL, HEALTH_VERSION, VIDEO_KEYS, seed_suite, task_seeds,
                               selected_episodes, partitions, check_observation, validate_result)


def observation(value=70):
    return {**{key: np.full((1, 8, 8, 3), value, dtype=np.uint8) for key in VIDEO_KEYS},
            'state.test': np.ones((1, 16), dtype=np.float32),
            'annotation.human.task_description': np.asarray(['test instruction'])}


class SeedOnlyTests(unittest.TestCase):
    def identity(self):
        return {'variant': 'baseline', 'checkpoint_sha256': 'a'*64, 'config_sha256': 'b'*64,
                'inference_backend': 'cuda-graph-pipeline', 'inference_backend_version': 'cpu_gpu_pipeline_batch1_v1',
                'scene_seed_definition_sha256': seed_suite()['scene_seed_definition_sha256'],
                'episode_protocol': PROTOCOL, 'n_envs': 4, 'evaluation_kind': 'formal'}

    def result(self, task='CloseBlenderLid', ids=None, diagnostic=False, max_replans=None):
        ids = list(range(50)) if ids is None else ids
        spec, seeds = task_seeds(task)
        identity = self.identity()
        requests = max_replans or spec['horizon']//5
        completed = max_replans is None
        rows = [{'episode': i, 'scene_seed': seeds[i], 'success': bool(i % 2), 'episode_complete': completed,
                 'requests': requests, 'health_checks': requests+1, 'initial_health': check_observation(observation())} for i in ids]
        successes = [r['success'] for r in rows] if completed else None
        metadata = {**identity, 'strict_full_state': {'all_loaded_values_exact_at_model_dtype': True},
                    'policy_rng_mode': 'stateless_request_v1', 'policy_seed_version': 'sha256_u32_v1'}
        return {'complete': completed, 'evaluation_kind': 'diagnostic' if diagnostic else 'formal',
                'env': 'robocasa/'+task, 'episode_protocol': PROTOCOL,
                'scene_seed_definition_sha256': identity['scene_seed_definition_sha256'],
                'episode_ids': ids, 'scene_seeds': [seeds[i] for i in ids], 'num_trials': len(ids), 'n_envs': 4,
                'logical_environment_index': 0, 'seed': 7, 'n_action_steps': 5,
                'max_episode_steps': spec['horizon'], 'diagnostic_max_replans': max_replans,
                'reference_plan_required': False, 'reference_observation_comparison': False,
                'observation_health_version': HEALTH_VERSION, 'episodes': rows,
                'successes': successes, 'num_successes': sum(successes) if completed else None,
                'requests': requests*len(ids), 'policy_server_metadata': metadata,
                'simulator_versions': {'robocasa': '1.0.0', 'robosuite': '1.5.2', 'egl_lifecycle': egl_lifecycle.VERSION}}

    def test_seed_and_episode_coverage_is_independent_of_worker_count(self):
        self.assertFalse(seed_suite()['reference_plan_required'])
        for task in TASKS:
            spec, seeds = task_seeds(task['task'])
            self.assertEqual(spec['horizon'], task['horizon'])
            self.assertEqual(len(seeds), 50)
            for count in range(1, 51):
                assignments = [(i, seeds[i]) for group in partitions(list(range(50)), count) for i in group]
                self.assertEqual(sorted(assignments), list(enumerate(seeds)))
        self.assertEqual(selected_episodes('0,3,49'), [0,3,49])
        for bad in ('', '0,0', '3,0', '-1', '50', 'one'):
            with self.assertRaises(ValueError): selected_episodes(bad)
        with self.assertRaises(ValueError): partitions([0,3], 4)

    def test_rgb_differences_are_allowed_but_unusable_frames_and_states_fail(self):
        left = observation(); right = observation()
        right[VIDEO_KEYS[0]][0, 0, 0, 0] = 145
        check_observation(left); check_observation(right)
        for mutation in (lambda d: d.pop(VIDEO_KEYS[0]),
                         lambda d: d.update({VIDEO_KEYS[1]: np.zeros((1,8,8,3),dtype=np.uint8)}),
                         lambda d: d.update({VIDEO_KEYS[0]: np.ones((8,8,4))}),
                         lambda d: d.update({VIDEO_KEYS[0]: np.full((8,8,3),np.nan)}),
                         lambda d: d.update({'state.test': np.asarray([np.inf])})):
            value = observation(); mutation(value)
            with self.assertRaises(ValueError): check_observation(value)

    def test_result_validation_rejects_missing_episodes_wrong_seed_and_partial_scores(self):
        data = self.result(); identity = self.identity()
        validate_result(data, 'CloseBlenderLid', list(range(50)), 4, identity)
        mutations = (lambda d: d['episodes'].pop(),
                     lambda d: d['episodes'][3].update(scene_seed=7),
                     lambda d: d['episodes'][3].update(episode_complete=False),
                     lambda d: d['episodes'][3].update(health_checks=0),
                     lambda d: d.update(max_episode_steps=1350),
                     lambda d: d.update(reference_plan_required=True),
                     lambda d: d['policy_server_metadata'].update(checkpoint_sha256='c'*64))
        for mutation in mutations:
            bad=copy.deepcopy(data); mutation(bad)
            with self.assertRaises(ValueError): validate_result(bad,'CloseBlenderLid',list(range(50)),4,identity)
        prefix = self.result(ids=[0,1,2,3],diagnostic=True,max_replans=8)
        validate_result(prefix,'CloseBlenderLid',[0,1,2,3],4,identity,True,8)
        self.assertIsNone(prefix['num_successes'])
        prefix['num_successes']=0
        with self.assertRaises(ValueError): validate_result(prefix,'CloseBlenderLid',[0,1,2,3],4,identity,True,8)

    def test_eight_gpu_scheduler_covers_exactly_50_tasks_and_one_live_slot_per_gpu(self):
        args=SimpleNamespace(part=None,gpu_ids=','.join(map(str,range(8))),egl_device_ids=None,variant='baseline',
                             home='home',config='config',checkpoint='weights',sim_python='sim',output='out',
                             n_envs=4,port_base=19090,timeout=14400)
        groups=evaluate_seeded.suite_commands(args)
        self.assertEqual(len(groups),8)
        names=[]
        for slot,commands in enumerate(groups):
            for command in commands:
                self.assertEqual(command[command.index('--gpu-ids')+1],str(slot))
                self.assertNotIn('--plan-dir',command)
                names.append(command[command.index('--task')+1])
        self.assertEqual(sorted(names),sorted(t['task'] for t in TASKS))
        self.assertEqual(len(names),len(set(names)))

    def test_native_child_exit_is_preserved_before_sibling_cleanup(self):
        class Child:
            instances=[]
            def __init__(self,**kwargs):
                self.slot=len(self.instances);self.pid=100+self.slot;self.exitcode=-int(signal.SIGABRT) if self.slot==0 else None
                self.instances.append(self)
            def start(self):pass
            def is_alive(self):return self.exitcode is None
            def terminate(self):self.exitcode=-15
            def join(self,timeout=None):pass
            def kill(self):self.exitcode=-9
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'native-failure'
            args=SimpleNamespace(diagnostic_episodes='0,3',diagnostic_max_replans=8,n_envs=2,
                                 task='CloseBlenderLid',output=output,host='localhost',port=1,timeout=60)
            with patch.object(seeded_episode_sim.mp,'get_context',return_value=SimpleNamespace(Process=Child)):
                with self.assertRaisesRegex(RuntimeError,'worker failed'):seeded_episode_sim.run(args)
            first=json.loads((output/'WORKER_FAILURE.json').read_text())['workers_before_cleanup']
            final=json.loads((output/'worker_exit.json').read_text())['after_cleanup']
            self.assertEqual(first[0]['exitcode'],-int(signal.SIGABRT));self.assertEqual(first[0]['signal'],'SIGABRT')
            self.assertIsNone(first[1]['exitcode']);self.assertEqual(final[1]['signal'],'SIGTERM')
            self.assertFalse((output/'result.json').exists())

    def test_actual_worker_uses_global_episode_without_trace_hashers_or_plan_checks(self):
        calls=[]; closed=[]
        class Policy:
            def __init__(self,**kwargs):
                self.assert_trace=kwargs['trace_determinism'];self._episode_indices=[];self._replan_indices=[];self._trace_hashers=[]
                self.server_metadata={'variant':'fake'}
                assert self.assert_trace is False
            def _ensure_counters(self,size):
                assert size==1
                if not self._episode_indices:self._episode_indices=[0];self._replan_indices=[0]
            def step(self,obs):
                assert obs[VIDEO_KEYS[0]].shape==(1,1,8,8,3)
                calls.append((self._episode_indices[0],self._replan_indices[0]));self._replan_indices[0]+=1
                return {'actions':{'test':np.ones((1,5,12))}}
            def finish_step(self,done):
                if done[0]:self._episode_indices[0]+=1;self._replan_indices[0]=0
            def close(self):closed.append('policy')
        class Env:
            def __init__(self):self.steps=0
            def step(self,actions):
                assert actions['test'].shape==(5,12)
                self.steps+=1;return observation(70+self.steps),float(self.steps==2),False,self.steps==3,{}
            def close(self):closed.append('env')
        created=[]
        def create(task,horizon,seed):
            assert not gc.isenabled(), 'Native renderer must not be interrupted by automatic cyclic GC'
            created.append((task,horizon,seed));return Env(),observation(),{}
        name='examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface'
        module=types.ModuleType(name);module.PolicyWrapper=Policy
        modules={name:module,'robocasa':SimpleNamespace(__version__='1.0.0'),'robosuite':SimpleNamespace(__version__='1.5.2')}
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules),patch.object(egl_lifecycle,'install'),patch.object(egl_lifecycle,'assert_healthy'),patch.object(fresh_scene,'create_scene',side_effect=create),patch.object(fresh_scene,'verify_scene',side_effect=AssertionError('No plan comparison allowed')):
            output=Path(directory)
            original_gc = gc.isenabled()
            seeded_episode_sim.worker(0,[0,3],'CloseBlenderLid',output,'localhost',1,file_sha(DEFINITION),None)
            self.assertEqual(gc.isenabled(), original_gc)
            data=json.loads((output/'worker0.json').read_text())
            self.assertEqual([r['episode'] for r in data['rows']],[0,3])
            self.assertTrue(all(r['episode_complete'] for r in data['rows']))
            self.assertEqual(calls,[(0,0),(0,1),(0,2),(3,0),(3,1),(3,2)])
            _,seeds=task_seeds('CloseBlenderLid')
            self.assertEqual([r[2] for r in created],[seeds[0],seeds[3]])
            self.assertEqual(closed.count('env'),2)
            self.assertEqual(closed.count('policy'),1)
            self.assertFalse(list(output.glob('*plan*')))
            class BlackEnv(Env):
                def step(self,actions):return observation(0),0.0,False,False,{}
            failed=output/'failed';failed.mkdir()
            with patch.object(fresh_scene,'create_scene',return_value=(BlackEnv(),observation(),{})):
                with self.assertRaisesRegex(ValueError,'All-zero RGB'):
                    seeded_episode_sim.worker(0,[3],'CloseBlenderLid',failed,'localhost',1,file_sha(DEFINITION),None)
            self.assertEqual(gc.isenabled(), original_gc)
            self.assertFalse((failed/'worker0.json').exists())
            self.assertEqual(json.loads((failed/'worker0.error.json').read_text())['episode'],3)
            self.assertTrue((failed/'worker0.failed_observation.npz').exists())

    def test_summary_requires_all_2500_and_never_summarizes_diagnostics(self):
        identity=self.identity()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);args=SimpleNamespace(output=output)
            for task in TASKS:
                path=output/'formal'/(task['task']+'.json');atomic_json(path,self.result(task['task']))
                atomic_json(path.with_suffix('.identity.json'),{'identity':identity,'result_sha256':file_sha(path)})
            evaluate_seeded.summarize(args,identity)
            self.assertEqual(json.loads((output/'FINAL_ACCEPTED.json').read_text())['trials'],2500)
            (output/'formal'/('WeighIngredients.json')).unlink()
            with self.assertRaises(FileNotFoundError):evaluate_seeded.summarize(args,identity)
            with self.assertRaises(ValueError):evaluate_seeded.summarize(args,{**identity,'evaluation_kind':'diagnostic'})

    @unittest.skipUnless(sys.platform.startswith('linux'),'Real Linux writer locks')
    def test_public_entry_reaches_policy_without_any_plans_and_rejects_mixed_weights(self):
        class PolicyBoundary(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);home=root/'home';(home/'receipts').mkdir(parents=True)
            (home/'receipts/sim.freeze.txt').write_text('real-test-fixture\n')
            config=root/'config.yaml';config.write_text('portable:\n  variant: baseline\n')
            checkpoint=root/'weights';checkpoint.write_bytes(b'first')
            output=root/'out'
            command=['evaluate_seeded.py','--variant','baseline','--home',str(home),'--config',str(config),
                     '--checkpoint',str(checkpoint),'--sim-python',sys.executable,'--output',str(output),
                     '--task','CloseBlenderLid','--n-envs','4']
            with patch.object(sys,'argv',command),patch.object(evaluate_seeded,'activate',side_effect=PolicyBoundary):
                with self.assertRaises(PolicyBoundary):evaluate_seeded.main()
            self.assertFalse((output/'SMOKE_ACCEPTED.json').exists())
            self.assertFalse((output/'plans').exists())
            self.assertEqual(json.loads((output/'EVALUATION_IDENTITY.json').read_text())['episode_protocol'],PROTOCOL)
            checkpoint.write_bytes(b'second')
            with patch.object(sys,'argv',command),patch.object(evaluate_seeded,'activate') as launch:
                with self.assertRaisesRegex(ValueError,'identity differs'):evaluate_seeded.main()
                launch.assert_not_called()


if __name__=='__main__':unittest.main()
