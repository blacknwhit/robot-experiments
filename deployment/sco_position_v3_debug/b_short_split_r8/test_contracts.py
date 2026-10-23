from pathlib import Path
import tempfile,unittest
from contracts import select_training,check_recipe,validate_complete
class ResumeSafetyTests(unittest.TestCase):
    def test_missing_completed_save_marker_rejected_before_deep_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);called=[]
            def deep(*args):called.append(True);return 1000
            with self.assertRaisesRegex(ValueError,'completed-save marker'):validate_complete(path,{},deep)
            self.assertEqual(called,[])
            (path/'PORTABLE_COMPLETE.json').write_text('{}')
            self.assertEqual(validate_complete(path,{},deep),1000);self.assertEqual(called,[True])
    def test_existing_recipe_requires_full_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'portable_recipe.json').write_text('{}')
            def none(*args):raise RuntimeError('No full state')
            with self.assertRaises(RuntimeError):select_training(run,{},run/'claim',none)
    def test_claim_without_checkpoint_never_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);claim=run/'claim';claim.touch()
            with self.assertRaisesRegex(RuntimeError,'refuse silent'):select_training(run,{},claim,None)
    def test_resume_latest_full_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'portable_recipe.json').write_text('{}')
            choice=select_training(run,{},run/'claim',lambda *args:(run/'checkpoints/checkpoint-1000',[{'step':1200}]))
            self.assertTrue(choice['resume']);self.assertEqual(choice['step'],1000)
    def test_new_engineering_run_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.assertEqual(select_training(run,{},run/'claim',None)['step'],0)
if __name__=='__main__':unittest.main()
