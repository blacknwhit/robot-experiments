"""Physical-time labels, source-row preservation, stale GT/state rejection."""
import copy
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import activate
activate('pri4r_paper',overlay=False)
from starVLA.pri4r.reader import (point_window,PointTrackStore,validate_episode,
    PHYSICAL_CLOCK_POLICY,LEGACY_CLOCK_POLICY,digest_file)
from starVLA.pri4r.generate import validate_recorded_clocks
from configure import render
from checkpoint import choose_latest,canonical_recipe
from pri4r_contract import validate_gt_receipt,validate_recipe


def fixture(length=61, gap=True):
    dataset=np.arange(length,dtype=np.float64)*.05
    physical=dataset+19.5
    if gap:physical[1:]+=.05
    # Nonlinear motion distinguishes a real displacement from divide-by-two or interpolation.
    tracks=np.stack([physical**2,physical**3,np.sin(physical)],axis=-1)[:,None,:]
    tracks=np.concatenate([tracks,tracks+np.array([1.,2.,3.])],axis=1).astype(np.float32)
    return tracks,np.repeat(np.eye(4)[None],length,axis=0),dataset,physical


def read(values,start,**kw):
    xyz,pose,dataset,physical=values
    return point_window(xyz,pose,dataset,start,coordinate_frame='world',
        simulator_timestamps=physical,clock_policy=PHYSICAL_CLOCK_POLICY,**kw)


class PhysicalClockTests(unittest.TestCase):
    def test_leading_gap_masks_whole_horizon_but_keeps_current(self):
        values=fixture();before=[a.copy() for a in values]
        out=read(values,0,expected_timestamp=0.)
        self.assertFalse(out['point_valid'].any())
        self.assertTrue((out['point_delta']==0).all())
        np.testing.assert_array_equal(out['point_current'],values[0][0])
        self.assertTrue(out['point_current_valid'].all())
        self.assertEqual(out['point_meta']['clock_masked_steps'],50)
        for a,b in zip(values,before):np.testing.assert_array_equal(a,b)

    def test_every_regular_later_window_is_bitwise_unchanged(self):
        values=fixture()
        for start in range(1,len(values[0])):
            new=read(values,start,expected_timestamp=values[2][start])
            old=point_window(*values[:3],start,coordinate_frame='world')
            for key in ('point_current','point_delta','point_valid','point_current_valid'):
                np.testing.assert_array_equal(new[key],old[key])
            self.assertEqual(new['point_meta']['clock_masked_steps'],0)

    def test_regular_episode_first_window_is_not_removed(self):
        values=fixture(gap=False)
        out=read(values,0)
        self.assertTrue(out['point_valid'].all())
        np.testing.assert_array_equal(out['point_delta'],
            point_window(*values[:3],0,coordinate_frame='world')['point_delta'])

    def test_single_frame_two_frames_and_eof(self):
        for length,gap,expected in ((1,True,0),(2,True,0),(2,False,2)):
            out=read(fixture(length,gap),0)
            self.assertEqual(out['point_valid'].sum(),expected)
        self.assertFalse(read(fixture(),60)['point_valid'].any())

    def test_two_distinct_clocks_are_required_and_not_substituted(self):
        xyz,pose,ds,physical=fixture()
        with self.assertRaisesRegex(ValueError,'requires recorded'):
            point_window(xyz,pose,ds,0,clock_policy=PHYSICAL_CLOCK_POLICY)
        with self.assertRaisesRegex(ValueError,'training timestamp'):
            read((xyz,pose,ds,physical),1,expected_timestamp=float(physical[1]))
        bad=physical.copy();bad[3]=float('nan')
        with self.assertRaises(ValueError):read((xyz,pose,ds,bad),0)

    def test_source_validation_rejects_unknown_gaps(self):
        _,_,ds,physical=fixture()
        self.assertEqual(validate_recorded_clocks(physical,ds,20,1e-5)['non_nominal_interval_indices'],[0])
        for mode in ('interior','first3dt','reverse'):
            bad=physical.copy()
            if mode=='interior':bad[3:]+=.05
            elif mode=='first3dt':bad[1:]+=.05
            else:bad[3]=bad[2]
            with self.assertRaises(ValueError):validate_recorded_clocks(bad,ds,20,1e-5)

    def test_cache_requires_physical_array_and_matching_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'task/episode_000000';path.mkdir(parents=True)
            xyz,poses,ds,physical=fixture(4)
            arrays={'world_tracks.npy':xyz,'base_poses.npy':poses,'timestamps.npy':ds,
                    'simulator_timestamps.npy':physical,'point_geom_ids.npy':np.array([1,2],np.int32),
                    'point_local.npy':np.zeros((2,3),np.float64),'point_face_ids.npy':np.array([0,1],np.int32),
                    'point_barycentric.npy':np.full((2,3),1/3,np.float64)}
            for name,a in arrays.items():np.save(path/name,a)
            meta={'schema':'pri4r_world_tracks_v1','complete':True,'frames':4,'points':2,
                  'episode_key':'task/episode_000000','cache_key':'test-clock-cache','config_sha256':'f'*64,
                  'fps':20.,'time_tolerance':1e-5,'clock_policy':PHYSICAL_CLOCK_POLICY,
                  'clock_diagnostic':validate_recorded_clocks(physical,ds,20.,1e-5),
                  'files':{name:{'sha256':digest_file(path/name),'bytes':(path/name).stat().st_size} for name in arrays}}
            def write(): (path/'complete.json').write_text(json.dumps(meta))
            write()
            store=PointTrackStore(tmp,num_points=2,coordinate_frame='world',clock_policy=PHYSICAL_CLOCK_POLICY)
            try:
                self.assertFalse(store.read('task',0,0,timestamp=0.)['point_valid'].any())
                self.assertTrue(store.read('task',0,1,timestamp=.05)['point_valid'][:,:2].all())
            finally:store.close()
            (path/'simulator_timestamps.npy').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_episode(path,require_physical_clock=True)
            np.save(path/'simulator_timestamps.npy',physical)
            bad=physical.copy();bad[3:]+=.05
            np.save(path/'simulator_timestamps.npy',bad)
            bad_store=PointTrackStore(tmp,num_points=2,clock_policy=PHYSICAL_CLOCK_POLICY)
            try:
                with self.assertRaisesRegex(ValueError,'interior simulator clock gap'):
                    bad_store.read('task',0,0)
            finally:bad_store.close()
            np.save(path/'simulator_timestamps.npy',physical)
            meta['clock_diagnostic']['initial_extra_dt_seconds']=0.;write()
            with self.assertRaisesRegex(ValueError,'receipt differs'):
                validate_episode(path,require_physical_clock=True)
            meta['clock_policy']=LEGACY_CLOCK_POLICY;write()
            with self.assertRaisesRegex(ValueError,'lacks the physical-clock'):
                validate_episode(path,require_physical_clock=True)

    def test_model_input_rejects_legacy_or_missing_clock_policy(self):
        source=ROOT/'runtimes/pri4r_paper/starVLA/model/framework/openpi/pi05.py'
        fn=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef)
                and x.name=='_validate_point_metadata')
        ns={'Any':object};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        cfg=SimpleNamespace(coordinate_frame='world',units='meters',horizon=50,num_points=1024,
                            output_kind='adjacent_displacement',clock_policy=PHYSICAL_CLOCK_POLICY)
        meta={'schema':'pri4r_world_tracks_v1',**vars(cfg)}
        ns['_validate_point_metadata'](meta,cfg)
        for missing in (True,False):
            bad=dict(meta)
            if missing:bad.pop('clock_policy')
            else:bad['clock_policy']=LEGACY_CLOCK_POLICY
            with self.assertRaises(ValueError):ns['_validate_point_metadata'](bad,cfg)

    def test_recipe_receipt_and_mixed_checkpoint_cannot_downgrade_policy(self):
        c=render('pri4r_paper','/tmp/clock-test','clock',1,2,smoke=True)
        broken=copy.deepcopy(c);broken['datasets']['vla_data']['point_track_clock_policy']=LEGACY_CLOCK_POLICY
        with self.assertRaisesRegex(ValueError,'physical-clock'):validate_recipe(broken)
        with self.assertRaisesRegex(ValueError,'Old GT receipt'):
            validate_gt_receipt(c,{'method':'pri4r_paper','verified_all_array_hashes':True},'/tmp/unused')
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for step,cfg in ((1,c),(2,broken)):
                cp=run/f'checkpoints/checkpoint-{step}';cp.mkdir(parents=True)
                (cp/'PORTABLE_COMPLETE.json').write_text(json.dumps({'step':step,'world':1,'recipe':canonical_recipe(cfg)}))
            chosen,rejected=choose_latest(run,c,validator=lambda *args:None)
            self.assertEqual(chosen.name,'checkpoint-1')
            self.assertEqual(rejected[0]['step'],2)
            self.assertIn('timing policy',rejected[0]['reason'])


if __name__=='__main__':unittest.main()
