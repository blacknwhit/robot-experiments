"""Test the real numerical gate without requiring CUDA or model weights."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class FakeFramework:
    def __init__(self):
        self.calls = 0
        self.offset = 0
    def predict_action(self, examples, **kwargs):
        self.calls += 1
        return {'actions': np.asarray(examples, dtype=np.float32)}


class FakeGraph:
    def __init__(self, framework):
        self.framework = framework
    def predict_action(self, examples, **kwargs):
        return {'actions': np.asarray(examples, dtype=np.float32) + self.framework.offset}


tree = ast.parse((ROOT / 'scripts/cuda_graph_policy.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CheckedCudaGraphPolicy')
namespace = {'np': np, 'CudaGraphPolicy': FakeGraph}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'cuda_graph_policy.py', 'exec'), namespace)
Checked = namespace['CheckedCudaGraphPolicy']


class GraphGateTests(unittest.TestCase):
    def test_smoke_checks_every_request_and_rejects_mismatch(self):
        framework = FakeFramework()
        policy = Checked(framework, verify_every_request=True)
        policy.predict_action([[1, 2]])
        policy.predict_action([[2, 3]])
        self.assertEqual(framework.calls, 2)
        framework.offset = 0.01
        with self.assertRaisesRegex(RuntimeError, 'differ from eager'):
            policy.predict_action([[3, 4]])
        self.assertEqual(policy.requests, 2)

    def test_formal_checks_first_request_then_reuses_verified_backend(self):
        framework = FakeFramework()
        policy = Checked(framework)
        policy.predict_action([[1, 2]])
        policy.predict_action([[3, 4]])
        self.assertEqual(framework.calls, 1)
        self.assertEqual(policy.requests, 2)

if __name__ == '__main__':
    unittest.main()
