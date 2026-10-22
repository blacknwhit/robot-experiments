import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fresh_scene
import fixed_episode_sim

SIM = 'examples.pi05_robocasa365.eval_files.simulation_env'
ADAPTER = 'examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface'


class FreshSceneTests(unittest.TestCase):
    def test_constructor_and_reset_are_independent_of_previous_random_consumption(self):
        closed = []
        def construct(args, index):
            self.assertEqual(index, 0)
            self.assertEqual(args.seed, 12345)
            before = (random.random(), np.random.rand())
            return SimpleNamespace(reset=lambda: ({'before': before, 'after': (random.random(), np.random.rand())}, {}),
                                   close=lambda: closed.append(True))
        with patch.dict(sys.modules, {SIM: SimpleNamespace(create_single_env=construct)}):
            a, obs_a, _ = fresh_scene.create_scene('robocasa/Test', 900, 12345)
            random.random(); np.random.rand(1000)
            b, obs_b, _ = fresh_scene.create_scene('robocasa/Test', 900, 12345)
            self.assertEqual(obs_a, obs_b)
            a.close(); b.close()
        self.assertEqual(len(closed), 2)

    def test_failed_first_reset_closes_constructed_environment(self):
        closed = []
        def reset(): raise RuntimeError('reset failed')
        with patch.dict(sys.modules, {SIM: SimpleNamespace(create_single_env=lambda *_:
                            SimpleNamespace(reset=reset, close=lambda: closed.append(True)))}):
            with self.assertRaisesRegex(RuntimeError, 'reset failed'):
                fresh_scene.create_scene('robocasa/Test', 900, 1)
        self.assertEqual(closed, [True])

    def test_batch_axes_and_dtypes_preserved(self):
        obs = {'video.a': np.zeros((1, 8, 8, 3), dtype=np.uint8),
               'state.a': np.zeros((1, 6), dtype=np.float64), 'annotation.a': 'test'}
        batched = fresh_scene.batch_observation(obs)
        self.assertEqual(batched['video.a'].shape, (1, 1, 8, 8, 3))
        self.assertEqual(batched['state.a'].dtype, np.float64)
        self.assertEqual(batched['annotation.a'].tolist(), ['test'])
        action = {'action.a': np.arange(30, dtype=np.float32).reshape(1, 5, 6)}
        np.testing.assert_array_equal(fresh_scene.unbatch_action(action)['action.a'], action['action.a'][0])
        with self.assertRaises(ValueError):fresh_scene.unbatch_action({'a': np.zeros((2, 5, 6))})

    def test_fresh_worker_full_horizon_success_trace_and_no_extra_reset(self):
        environments = []; policies = []
        class Env:
            def __init__(self, seed): self.seed=seed; self.steps=0; self.closed=False
            def step(self, actions):
                self.steps+=1
                self_check.assertEqual(actions['a'].shape, (5, 6))
                return {'state.a': np.zeros((1, 6))}, self.steps==1, False, self.steps==2, {}
            def close(self):self.closed=True
        class Policy:
            def __init__(self, **kwargs):
                self._episode_indices=[0]; self._replan_indices=[0]; self._trace_hashers=[None]
                self.episode_trace_hashes=[]; self.server_metadata={'seed': kwargs['base_seed']}
                self.done=[]; self.closed=False; policies.append(self)
            def _ensure_counters(self, n):pass
            def _new_trace_hasher(self, i):return None
            def step(self, obs):
                self_check.assertEqual(obs['state.a'].shape, (1, 1, 6))
                return {'actions': {'a': np.zeros((1, 5, 6), dtype=np.float32)}}
            def finish_step(self, done):
                self.done.append(bool(done[0]))
                if done[0]:self.episode_trace_hashes.append(str(self._episode_indices[0]))
            def close(self):self.closed=True
        def create(task, horizon, seed):
            env=Env(seed); environments.append(env)
            return env, {'state.a': np.zeros((1, 6))}, {}
        self_check=self
        plan={'seed':7, 'env':'robocasa/Test', 'max_episode_steps':10, 'robocasa_version':'1',
              'robosuite_version':'2', 'resets':[{'scene_seed':i} for i in range(9)]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                ADAPTER:SimpleNamespace(PolicyWrapper=Policy), 'robocasa':SimpleNamespace(__version__='1'),
                'robosuite':SimpleNamespace(__version__='2')}), patch('fresh_scene.create_scene', side_effect=create), \
                patch('fresh_scene.fingerprints', return_value={}), patch('fresh_scene.verify_scene'), \
                patch('fixed_episode_sim.file_sha', return_value='sha'):
            output=Path(tmp)
            fixed_episode_sim.replay_fresh_worker(0,[7,0],plan,output/'plan','sha',output,'localhost',0,None)
            result=json.loads((output/'worker0.json').read_text())
            self.assertEqual([r['episode'] for r in result['rows']], [7,0])
            self.assertEqual([r['requests'] for r in result['rows']], [2,2])
            self.assertEqual([r['success'] for r in result['rows']], [True,True])
            self.assertEqual([r['trace'] for r in result['rows']], ['7','0'])
            self.assertEqual([r['episode'] for r in result['initial_scenes']], [7,0])
        self.assertEqual([e.seed for e in environments],[7,0])
        self.assertTrue(all(e.closed for e in environments))
        self.assertTrue(policies[0].closed)
        self.assertEqual(policies[0].done,[False,True,False,True])


if __name__ == '__main__':unittest.main()
