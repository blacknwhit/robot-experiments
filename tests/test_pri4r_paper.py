"""Paper recipe, bounded data isolation, and meaningful auxiliary math checks."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from configure import render
from common import runtime,is_point_variant,sha256
from pri4r_contract import validate_recipe,validate_gt_receipt


class ContractTests(unittest.TestCase):
    def recipe(self):
        return render('pri4r_paper',Path(tempfile.gettempdir())/'pri4r-contract-test','smoke',4,2,smoke=True)

    def test_paper_recipe(self):
        c=self.recipe();validate_recipe(c)
        self.assertEqual(c['datasets']['vla_data']['per_device_batch_size'],32)
        self.assertEqual(c['trainer']['ema_decay'],0.999)
        self.assertEqual(runtime('pri4r_paper').name,'pri4r_paper')
        self.assertTrue(is_point_variant('pri4r_paper'))

    def test_method_or_recipe_changes_rejected(self):
        for field,value in [('ema_decay',None),('num_warmup_steps',1000),('max_train_steps',100000)]:
            c=self.recipe();c['trainer'][field]=value
            with self.assertRaises(ValueError):validate_recipe(c)
        c=self.recipe();c['framework']['pi05']['foresight']['enabled']=True
        with self.assertRaises(ValueError):validate_recipe(c)

    def test_smoke_cannot_be_formal(self):
        c=self.recipe();c['portable']['smoke']=False;c['portable']['formal']=True
        with self.assertRaises(ValueError):validate_recipe(c)
        with self.assertRaises(ValueError):render('pri4r_paper','x','x',4,30000,smoke=True)

    def test_bounded_gt_receipt(self):
        c=self.recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'input_manifest.json';manifest.write_text('{"episodes":[]}')
            c['datasets']['vla_data']['pri4r_smoke_manifest']=str(manifest)
            c['datasets']['vla_data']['point_track_gt_config_sha256']='f'*64
            proof={'method':'pri4r_paper','bounded':True,'episodes':4,'frames':1126,
                   'verified_all_array_hashes':True,'config_sha256':'f'*64,'manifest_sha256':sha256(manifest)}
            validate_gt_receipt(c,proof,root)
            proof['bounded']=False
            with self.assertRaises(ValueError):validate_gt_receipt(c,proof,root)

    def test_no_contact_sampling_rules(self):
        cfg=json.loads((ROOT/'configs/pri4r_paper_gt.json').read_text())
        self.assertNotIn('sampling_rules',cfg)
        self.assertEqual(cfg['num_points'],1024)


if __name__=='__main__':unittest.main()
