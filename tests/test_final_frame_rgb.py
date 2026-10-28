"""Preserve control count and the consumed frame, including a partial final chunk."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import final_frame_rgb


class FinalFrameTests(unittest.TestCase):
    def test_interleaved_envs_partial_chunk_and_cleanup(self):
        sentinels=[object(),object(),object()]
        class Raw:
            def __init__(self):
                self.tick=0;self.camera_names=['left','right','wrist'];self.camera_heights=self.camera_widths=2
                self.ignore_done=True;self.camera_depths=[False]*3;self.camera_segmentations=[None]*3
                self.control_timestep=.05;self._observables={};self.env=self;self.unwrapped=self
                for name in self.camera_names:
                    def sensor(cache,raw=self):return np.full((2,2,3),raw.tick,dtype=np.uint8)
                    sensor.__modality__='image'
                    self._observables[name+'_image']=NS(_sensor=sensor,_current_observed_value=np.zeros((2,2,3),dtype=np.uint8),
                        _sampling_timestep=.05,_corrupter=sentinels[0],_filter=sentinels[1],_delayer=sentinels[2],
                        is_enabled=lambda:True,is_active=lambda:True)
            def step(self,action):
                self.tick+=1
                for o in self._observables.values():o._current_observed_value=o._sensor({})
                return self.tick
        class Multi:
            def __init__(self,raw):
                self.env=raw;self.n_action_steps=5;self.max_episode_steps=7;self.reward=[];self.done=[]
                self.video_delta_indices=self.state_delta_indices=[0];self.unwrapped=raw
            def step(self,action):
                for _ in range(min(self.n_action_steps,self.max_episode_steps-len(self.reward))):
                    self.env.step(action);self.reward.append(0)
                return self.env.tick
        modules={'robocasa.wrappers.gym_wrapper':NS(RoboCasaGymEnv=Raw),
                 'examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper':NS(MultiStepWrapper=Multi),
                 'robosuite.utils.observables':NS(NO_CORRUPTION=sentinels[0],NO_FILTER=sentinels[1],NO_DELAY=sentinels[2])}
        old_raw,old_multi=Raw.step,Multi.step
        with patch.dict(sys.modules,modules):
            report=final_frame_rgb.install();references=[]
            for wave in range(2):
                envs=[Multi(Raw()) for _ in range(4)]
                for _ in range(2):
                    for env in envs:
                        result=env.step(None)
                        for observable in env.env._observables.values():
                            self.assertTrue(np.all(observable._current_observed_value==result))
                for env in envs:
                    self.assertEqual(env.env.tick,7)
                    references.append(weakref.ref(env.env));report.release(env)
                del env,envs,observable
                gc.collect()
                self.assertTrue(all(ref() is None for ref in references))
            self.assertEqual(report()['control_steps'],56)
            self.assertEqual(report()['rgb_rendered'],48)
            self.assertEqual(report()['rgb_reused'],120)
            self.assertEqual(report.active_count(),0)
            report.close()
        self.assertIs(Raw.step,old_raw);self.assertIs(Multi.step,old_multi)


if __name__=='__main__':unittest.main()
