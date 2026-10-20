"""CPU oracles for the authorized position/temporal experiments; no model downloads."""
from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtimes/trajectory_v3"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "playground/Code/openpi_robocasa365/src"), str(ROOT / "scripts")]
AVAILABLE = all(importlib.util.find_spec(x) for x in ("torch", "numpy", "yaml", "omegaconf"))
if AVAILABLE:
    import numpy as np
    import torch
    from openpi.models_pytorch.point_track import PointTrackConfig, distributed_point_track_loss
    from openpi.models_pytorch.position_point import PositionPointHead
    from openpi.models_pytorch.position_supervision import position_loss_terms, position_diagnostic_sums, diagnostic_names
    from starVLA.pri4r.reader import point_window
    from starVLA.training.trajectory_contract import write_checkpoint_contract, validate_contract
    from configure import render
    from omegaconf import OmegaConf
    torch.set_num_threads(1)


def config(layers=0, **kw):
    values = dict(enabled=True, version="pri4r_embedding_v1", horizon=5, num_points=3,
                  hidden_dim=16, num_heads=2, point_hidden_dim=8, fusion_hidden_dim=8,
                  objective_version="relative_position_and_difference_v3", output_kind="start_relative_position",
                  trajectory_horizons=(2, 5), trajectory_loss_weight=0., auxiliary_warmup_steps=1000,
                  temporal_layers=layers, temporal_heads=2, temporal_ff_dim=16, temporal_rope=False,
                  point_chunk_size=2, checkpoint_fusion=True)
    values.update(kw)
    return PointTrackConfig(**values)


@unittest.skipUnless(AVAILABLE, "CPU torch/numpy/yaml/omegaconf required")
class PositionV3Tests(unittest.TestCase):
    def test_reader_fixed_coordinate_anchor_eof_and_gap(self):
        tracks = np.zeros((6, 3, 3), np.float32)
        tracks[:, :, 0] = 10 + np.arange(6)[:, None] * .01
        poses = np.tile(np.eye(4), (6, 1, 1))
        poses[:, :3, 3] = np.arange(6)[:, None] * 99.  # future poses must NOT set coordinates
        poses[0, 0, 3] = 10
        poses[0, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        times = np.arange(6) * .05
        out = point_window(tracks, poses, times, 0, horizon=8, output_kind="start_relative_position")
        self.assertNotIn("point_delta", out)
        np.testing.assert_allclose(out["point_position"][0, :5, 1], -np.arange(1, 6) * .01, atol=1e-6)
        self.assertTrue(out["point_valid"][:, :5].all())
        self.assertFalse(out["point_valid"][:, 5:].any())
        self.assertEqual(float(np.abs(out["point_position"][:, 5:]).sum()), 0.)
        # A point-specific missing frame invalidates its entire suffix.
        tracks[2, 0] = np.nan
        gap = point_window(tracks, poses, times, 0, output_kind="start_relative_position")
        self.assertTrue(gap["point_valid"][0, 0])
        self.assertFalse(gap["point_valid"][0, 1:].any())
        self.assertTrue(gap["point_valid"][1, :5].all())
        self.assertTrue(np.isfinite(gap["point_position"]).all())
        self.assertEqual(gap["point_meta"]["position_validity"], "complete_prefix_v1")

    def test_static_truth_remains_valid_and_perfect_track_has_zero_loss(self):
        tracks = np.ones((6, 3, 3), np.float32)
        out = point_window(tracks, np.tile(np.eye(4), (6, 1, 1)), np.arange(6) * .05,
                           0, horizon=5, output_kind="start_relative_position")
        self.assertTrue(out["point_valid"].all())
        target = torch.from_numpy(out["point_position"])[None]
        p = target.clone().requires_grad_()
        ps, pc, ds, dc = position_loss_terms(p, target, torch.from_numpy(out["point_valid"])[None])
        self.assertEqual((ps.item(), pc.item(), ds.item(), dc.item()), (0., 45., 0., 45.))
        (ps + ds).backward()
        self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_dense_position_error_has_no_horizon_division(self):
        p = (torch.arange(1, 51) * .0005).view(1, 1, 50, 1).expand(-1, -1, -1, 3).clone().requires_grad_()
        ps, pc, ds, dc = position_loss_terms(p, torch.zeros_like(p), torch.ones(1, 1, 50, dtype=torch.bool))
        self.assertAlmostEqual(float((ps / pc).detach()), .01275, places=7)
        self.assertAlmostEqual(float((ds / dc).detach()), .0005, places=7)
        (ps / pc + ds / dc).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_nan_zero_connection_and_reject_nonprefix(self):
        p = torch.randn(1, 2, 5, 3, requires_grad=True)
        y = torch.full_like(p, float("nan"))
        v = torch.zeros(1, 2, 5, dtype=torch.bool)
        ps, pc, ds, dc = position_loss_terms(p, y, v)
        self.assertEqual(float((ps + pc + ds + dc).detach()), 0.)
        (ps + ds).backward()
        self.assertEqual(float(p.grad.abs().sum()), 0.)
        v[:, :, 1] = True
        with self.assertRaises(ValueError):
            position_loss_terms(p, torch.zeros_like(p), v)
        v.fill_(True)
        with self.assertRaises(FloatingPointError):
            position_loss_terms(p, y, v)
        with self.assertRaises(FloatingPointError):
            position_loss_terms(torch.full_like(p, float("nan")), p, torch.zeros_like(v))

    def test_numpy_gradient_oracle_unequal_rank_masks(self):
        rng = np.random.default_rng(20261020)
        for world, acc in ((2, 1), (2, 2), (8, 1)):
            x = rng.normal(size=(world * acc, 2, 5, 3)).astype(np.float32)
            y = rng.normal(size=x.shape).astype(np.float32)
            v = np.arange(5)[None, None] < np.arange(world * acc)[:, None, None] % 6
            v = np.broadcast_to(v, x.shape[:-1]).copy()
            count = v.sum() * 3
            expected = np.zeros_like(x)
            total = 0.
            for i in range(len(x)):
                for n in range(2):
                    for h in range(5):
                        if not v[i, n, h]:
                            continue
                        e = x[i, n, h] - y[i, n, h]
                        expected[i, n, h] += np.sign(e) / max(count, 1)
                        de = e - (x[i, n, h-1] - y[i, n, h-1] if h else 0)
                        expected[i, n, h] += np.sign(de) / max(count, 1)
                        if h:
                            expected[i, n, h-1] -= np.sign(de) / max(count, 1)
                        total += (np.abs(e).sum() + np.abs(de).sum()) / max(count, 1)
            p = torch.tensor(x, requires_grad=True)
            value = p.sum() * 0
            for i in range(len(x)):
                ps, _, ds, _ = position_loss_terms(p[i:i+1], torch.tensor(y[i:i+1]), torch.tensor(v[i:i+1]))
                local = distributed_point_track_loss(torch.stack((ps, ds)), torch.tensor([count, count]),
                                                    world_size=world, accumulation_steps=acc).sum()
                value = value + local / acc / world
            value.backward()
            np.testing.assert_allclose(p.grad.numpy(), expected, rtol=2e-6, atol=1e-8)
            self.assertAlmostEqual(float(value.detach()), total, places=5)

    def test_causal_direction_and_zero_future_gradient_at_decoder_boundary(self):
        for training in (True, False):
            head = PositionPointHead(16, config(2)).train(training)
            points = torch.randn(1, 3, 8)
            time = torch.randn(1, 5, 8, requires_grad=True)
            first = head._fuse_projected(points, time)
            changed = time.detach().clone()
            changed[:, 2:] += 100 * torch.randn_like(changed[:, 2:])
            with torch.no_grad():
                second = head._fuse_projected(points, changed)
            torch.testing.assert_close(first[:, :, :2], second[:, :, :2], atol=1e-6, rtol=1e-5)
            first[:, :, :2].sum().backward()
            self.assertEqual(float(time.grad[:, 2:].abs().sum()), 0.)
            self.assertGreater(float(time.grad[:, :2].abs().sum()), 0.)

    def test_common_initialization_rng_and_chunk_gradient_equivalence(self):
        torch.manual_seed(42)
        a = PositionPointHead(16, config(0)); state_a = torch.get_rng_state()
        torch.manual_seed(42)
        b = PositionPointHead(16, config(2)); state_b = torch.get_rng_state()
        self.assertTrue(torch.equal(state_a, state_b))
        for name, value in a.named_parameters():
            self.assertTrue(torch.equal(value, dict(b.named_parameters())[name]), name)
        memory = torch.randn(2, 4, 16)
        valid = torch.ones(2, 4, dtype=torch.bool)
        points = torch.randn(2, 3, 3)
        for layers in (0, 2):
            chunked = PositionPointHead(16, config(layers))
            whole = PositionPointHead(16, config(layers, point_chunk_size=3, checkpoint_fusion=False))
            whole.load_state_dict(chunked.state_dict())
            outputs = []
            for head in (chunked, whole):
                h = memory.clone().requires_grad_()
                out = head(h, valid, points)
                out.square().sum().backward()
                outputs.append((out.detach(), h.grad))
                for name, param in head.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertGreater(float(param.grad.abs().sum()), 0., name)
            for x, y in zip(*outputs):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)
            for (_, x), (_, y) in zip(chunked.named_parameters(), whole.named_parameters()):
                torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-4)
            permutation = torch.tensor([2, 0, 1])
            with torch.no_grad():
                torch.testing.assert_close(whole(memory, valid, points[:, permutation]),
                                           whole(memory, valid, points)[:, permutation], atol=1e-6, rtol=1e-5)

    def test_signatures_reject_same_shape_semantic_change_even_non_strict(self):
        head = PositionPointHead(16, config(0))
        old = replace(config(0), objective_version="adjacent_and_cumulative_v2",
                      output_kind="adjacent_displacement", trajectory_loss_weight=1.)
        state = copy.deepcopy(head.state_dict())
        state["point_track_signature"] = old.signature(16)
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            head.load_state_dict(state, strict=False)
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            PositionPointHead(16, config(2)).load_state_dict(head.state_dict(), strict=False)
        with self.assertRaises(ValueError):
            config(2, temporal_rope=True).validate()

    def test_cpu_adam_checkpoint_rng_roundtrip_next_update_exact(self):
        torch.manual_seed(111)
        head = PositionPointHead(16, config(2))
        optimizer = torch.optim.AdamW(head.parameters(), lr=2.5e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        def step(model, opt, sched):
            memory, points = torch.randn(2, 4, 16), torch.randn(2, 3, 3)
            y = torch.randn(2, 3, 5, 3)
            p = model(memory, torch.ones(2, 4, dtype=torch.bool), points)
            ps, pc, ds, dc = position_loss_terms(p, y, torch.ones(2, 3, 5, dtype=torch.bool))
            loss = ps / pc + ds / dc
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            return loss.detach()
        for _ in range(2):
            step(head, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(dict(model=head.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), rng=torch.get_rng_state()), path)
            expected_loss = step(head, optimizer, scheduler)
            restored = PositionPointHead(16, config(2))
            opt = torch.optim.AdamW(restored.parameters(), lr=2.5e-5)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 / (s + 1))
            state = torch.load(path, weights_only=True)
            restored.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"]); torch.set_rng_state(state["rng"])
            self.assertTrue(torch.equal(step(restored, opt, sched), expected_loss))
            for (name, p), (_, q) in zip(head.named_parameters(), restored.named_parameters()):
                self.assertTrue(torch.equal(p, q), name)
            self.assertEqual(scheduler.state_dict(), sched.state_dict())
            for a, b in zip(optimizer.state.values(), opt.state.values()):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(a[key], b[key]))

    def test_position_diagnostics_reveal_static_drift(self):
        p = torch.ones(1, 2, 5, 3, requires_grad=True) * .01
        sums = position_diagnostic_sums(p, torch.zeros_like(p), torch.ones(1, 2, 5, dtype=torch.bool),
                                        torch.tensor([[1, 2]]), (2, 5))
        values = dict(zip(diagnostic_names((2, 5)), sums.tolist()))
        self.assertFalse(sums.requires_grad)
        self.assertGreater(values["position_diag/fde5_object_path_le_10mm_sum"], .017)
        self.assertEqual(values["position_diag/fde5_object_path_le_10mm_zero_sum"], 0.)

    def test_portable_engineering_bounds_and_resume_contract(self):
        for name, layers in (("embedding3d_pos_a", 0), ("embedding3d_temporal_b", 2)):
            with self.assertRaises(ValueError):
                render(name, "/tmp/v3", "formal", 2)
            with self.assertRaises(ValueError):
                render(name, "/tmp/v3", "long", 2, stop=1501, engineering=True)
            for world in (2, 4, 8):
                c = render(name, "/tmp/v3", "short", world, stop=1500, engineering=True)
                head = PointTrackConfig.parse(c["framework"]["pi05"]["point_track"])
                self.assertEqual(head.temporal_layers, layers)
                self.assertEqual(c["datasets"]["vla_data"]["per_device_batch_size"] * world, 64)
                self.assertEqual(c["trainer"]["max_train_steps"], 100000)
                self.assertEqual(c["trainer"]["save_interval"], 1000)
        with tempfile.TemporaryDirectory() as tmp:
            c = render("embedding3d_pos_a", tmp, "smoke", 2, stop=2, smoke=True)
            c["output_dir"] = tmp
            cp = Path(tmp) / "checkpoints/checkpoint-2"; cp.mkdir(parents=True)
            write_checkpoint_contract(OmegaConf.create(c), 2)
            validate_contract(cp / "trajectory_contract.json", c, 2)
            changed = copy.deepcopy(c); changed["framework"]["pi05"]["point_track"]["temporal_layers"] = 2
            with self.assertRaises(ValueError):
                validate_contract(cp / "trajectory_contract.json", changed, 2)


if __name__ == "__main__":
    unittest.main()
