import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from focalvla.common import lr_at
from focalvla.data import PlanBatches
from focalvla.checkpoint import seed, rng_state, restore_rng, save, restore, latest


class ProtocolTest(unittest.TestCase):
    def test_schedule(self):
        c = dict(warmup_steps=10000,total_steps=70000,peak_lr=5e-5,final_lr=5e-6)
        self.assertAlmostEqual(lr_at(9999,c),5e-5)
        self.assertAlmostEqual(lr_at(69999,c),5e-6)
        self.assertGreater(lr_at(10000,c),lr_at(50000,c))

    def test_eight_rank_partition_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            draws = np.arange(5*64).reshape(5,64); np.save(Path(tmp)/'draws.npy',draws)
            for step in (0,3,4):
                ranks = [next(iter(PlanBatches(tmp,step,step+1,rank,8,8,1))) for rank in range(8)]
                np.testing.assert_array_equal(np.concatenate(ranks),draws[step])

    def test_partial_and_incompatible(self):
        from focalvla.common import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'step_000001.tmp').write_bytes(b'incomplete')
            self.assertEqual(latest(root,'same'),(0,None))
            (root/'step_000002.pt').write_bytes(b'valid')
            atomic_json(root/'step_000002.json',dict(file='step_000002.pt',bytes=5,step=2,contract='other'))
            with self.assertRaises(ValueError): latest(root,'same')

    def test_exact_continuation(self):
        seed(42); model=torch.nn.Linear(3,2); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        draws=np.arange(4*2).reshape(4,2)
        def update():
            optimizer.zero_grad(); model(torch.randn(2,3)).square().mean().backward(); optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as tmp:
            path=save(tmp,model,optimizer,1,'test',[rng_state()],draws,2,.01)
            update(); expected={k:v.clone() for k,v in model.state_dict().items()}
            following=torch.randn(5)
            restore(path,model,optimizer,'test',0,1,draws,2,lambda _: .01)
            update()
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            torch.testing.assert_close(torch.randn(5),following,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
