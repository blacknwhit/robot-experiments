import unittest
from allocator_environment import case_environment,ALLOCATOR_KEYS
class AllocatorEnvironmentTests(unittest.TestCase):
    def test_original_preserves_both_allocator_aliases(self):
        parent={k:'expandable_segments:True' for k in ALLOCATOR_KEYS}
        self.assertEqual(case_environment(parent,'a_original'),{**parent,'REFERENCE_EVAL_CASE':'a_original'})
    def test_candidate_removes_only_allocator_and_preserves_parent(self):
        parent={k:'expandable_segments:True' for k in ALLOCATOR_KEYS}
        parent.update(NVIDIA_DRIVER_CAPABILITIES='all',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='runtime',LD_PRELOAD='lib',OMP_NUM_THREADS='4')
        original=dict(parent)
        for case in ('a_default_allocator','baseline_reference'):
            expected={k:v for k,v in parent.items() if k not in ALLOCATOR_KEYS}
            self.assertEqual(case_environment(parent,case),{**expected,'REFERENCE_EVAL_CASE':case})
        self.assertEqual(parent,original)
    def test_unknown_case_refused(self):
        with self.assertRaises(ValueError):case_environment({},'train')
if __name__=='__main__':unittest.main()
