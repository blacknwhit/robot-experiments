import copy
import gc
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fresh_scene
import seeded_async_sim
import simulator_renderer as adapter
from seeded_evaluation import VIDEO_KEYS, EXECUTION_VERSION


def image_observation(state, black=False):
    return {**{key: np.full((1,8,8,3),0 if black else 90,dtype=np.uint8) for key in VIDEO_KEYS},
            'state.test': np.asarray([[state]],dtype=np.float64)}


class AsyncEnvironmentsTests(unittest.TestCase):
    def exercise(self, n_envs, directory, fail=False):
        traces=[];envs=[];clients=[];main=threading.get_ident()
        class Policy:
            def __init__(self,**kwargs):
                self._episode_indices=[];self._replan_indices=[];self.busy=False;self.closed=False
                self.server_metadata={'variant':'fixture'};clients.append(self)
                assert not kwargs['trace_determinism']
            def _ensure_counters(self,n):
                assert n==1
                if not self._episode_indices:self._episode_indices=[0];self._replan_indices=[0]
            def step(self,observation):
                assert threading.get_ident()!=main and not self.busy
                self.busy=True
                try:
                    ep=self._episode_indices[0];step=self._replan_indices[0]
                    value=float(observation['state.test'][0,0,0])
                    # Force different completion order without touching scene RNG.
                    time.sleep(.002*(4-ep))
                    traces.append((ep,step,value));self._replan_indices[0]+=1
                    return {'actions':{'action.test':np.full((1,5,1),value)}}
                finally:self.busy=False
            def finish_step(self,done):
                assert threading.get_ident()==main and not self.busy
                if done[0]:self._episode_indices[0]+=1;self._replan_indices[0]=0
            def close(self):self.closed=True
        class Env:
            def __init__(self):
                self.state=random.random()+float(np.random.random());self.steps=0;self.closed=False;envs.append(self)
            def step(self,actions):
                assert threading.get_ident()==main and not self.closed
                self.steps+=1
                self.state=random.random()+float(np.random.random())+float(actions['action.test'][0,0])*.1
                return image_observation(self.state,black=fail),float(self.state>1),False,self.steps==3,{}
            def close(self):
                assert threading.get_ident()==main;self.closed=True
        def create(task,horizon,seed):
            assert threading.get_ident()==main and not gc.isenabled()
            random.seed(seed);np.random.seed(seed);env=Env();return env,image_observation(env.state),{}
        def retain(environments):
            assert threading.get_ident()==main and all(not env.closed for env in environments)
            return [{'adapter':adapter.VERSION,'engine':'mujoco','backend':'egl','msaa':4,
                     'max_geom':10000,'gl_context_type':'mujoco.egl.GLContext'} for env in environments]
        name='examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface'
        module=ModuleType(name);module.PolicyWrapper=Policy
        modules={name:module,'robocasa':NS(__version__='1.0.0'),'robosuite':NS(__version__='1.5.2')}
        args=NS(task='CloseBlenderLid',output=Path(directory)/('n'+str(n_envs)),host='localhost',port=1,n_envs=n_envs,
                diagnostic_episodes='0,1,2,3',diagnostic_max_replans=None,timeout=60)
        before_gc=gc.isenabled()
        with patch('robot_eval.client.create_policy',side_effect=lambda **kw:Policy(trace_determinism=False,**kw)),patch.dict(sys.modules,modules),patch.object(adapter,'install'),patch.object(adapter,'assert_healthy'),patch.object(adapter,'retain_environments',side_effect=retain),patch.object(adapter,'release_all'),patch.object(fresh_scene,'create_scene',side_effect=create),patch.object(fresh_scene,'verify_scene',side_effect=AssertionError('No reference plans')):
            if fail:
                with self.assertRaisesRegex(ValueError,'All-zero RGB'):
                    seeded_async_sim.run(args)
                self.assertFalse((args.output/'result.json').exists())
                error=json.loads((args.output/'SIMULATION_ERROR.json').read_text())
                self.assertEqual(error['phase'],'simulation')
                result=None
            else:
                result=seeded_async_sim.run(args)
        self.assertEqual(gc.isenabled(),before_gc)
        self.assertTrue(all(env.closed for env in envs))
        self.assertTrue(all(client.closed and not client.busy for client in clients))
        return result,sorted(traces)

    def test_scene_rng_and_policy_requests_independent_of_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            one,trace1=self.exercise(1,directory)
            two,trace2=self.exercise(2,directory)
            four,trace4=self.exercise(4,directory)
        self.assertEqual(trace1,trace2)
        self.assertEqual(trace1,trace4)
        self.assertEqual(four['simulator_execution'],EXECUTION_VERSION)
        self.assertEqual(four['episode_ids'],[0,1,2,3])
        self.assertEqual(four['scene_seeds'],one['scene_seeds'])
        self.assertEqual(four['successes'],one['successes'])
        self.assertEqual(four['requests'],12)
        self.assertTrue(four['complete'])

    def test_failure_closes_all_live_envs_and_clients_without_score(self):
        with tempfile.TemporaryDirectory() as directory:self.exercise(4,directory,True)

    def test_renderer_retains_every_live_environment(self):
        class Context:
            _native_rgb_version=adapter.VERSION
            def __init__(self):
                self.con=NS(offSamples=4);self.scn=NS(maxgeom=10000);self.gl_ctx=object()
        first,second,stale=Context(),Context(),Context()
        def env(context):return NS(unwrapped=NS(env=NS(sim=NS(_render_context_offscreen=context))))
        with patch.object(adapter,'_contexts',{first,second,stale}),patch.object(adapter,'_release') as release:
            proof=adapter.retain_environments([env(first),env(second)])
            self.assertEqual(len(proof),2)
            release.assert_called_once_with(stale)


if __name__=='__main__':unittest.main()
