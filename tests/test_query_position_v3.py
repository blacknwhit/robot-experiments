"""Focused C wiring/decoder oracles. CPU fixtures do not certify a full GPU policy."""
import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from test_position_v3 import AVAILABLE, ROOT, RUNTIME, config

if AVAILABLE:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from openpi.models_pytorch.foresight import ForesightConfig
    from openpi.models_pytorch.position_point import (
        PositionPointHead, QueryPositionPointHead, validate_position_policy)
    from common import POSITION_QUERY_VARIANTS, runtime
    from configure import render
    from engineering_audit import audited_parameter_names
    from evaluate_frozen import child_commands

    # Execute the real production methods, replacing only the large transformer
    # and preprocessing with a small deterministic attention fixture.
    path = RUNTIME / 'playground/Code/openpi_robocasa365/src/openpi/models_pytorch/pi0_pytorch.py'
    source = ast.parse(path.read_text(encoding='utf-8'))
    selected = {'embed_prefix', 'forward', 'sample_actions', 'denoise_step'}
    methods = []
    for item in source.body:
        if isinstance(item, ast.FunctionDef) and item.name == 'make_att_2d_masks':
            methods.append(item)
        if isinstance(item, ast.ClassDef) and item.name == 'PI0Pytorch':
            methods.extend(x for x in item.body if isinstance(x, ast.FunctionDef) and x.name in selected)
    namespace = dict(torch=torch, F=F, Tensor=torch.Tensor, math=__import__('math'))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    make_mask = namespace['make_att_2d_masks']

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(16, 16)
            self.norm = nn.LayerNorm(16)
            self.paligemma = NS(language_model=NS(
                layers=[NS(self_attn=NS(q_proj=self.projection))], config=NS()))
            self.gemma_expert = NS(model=NS(config=NS()))

        def embed_language_tokens(self, tokens):
            return tokens

        def forward(self, *, inputs_embeds, attention_mask, past_key_values, **kwargs):
            prefix, suffix = inputs_embeds
            x = torch.cat([v for v in (prefix, suffix) if v is not None], dim=1)
            q = self.projection(x)
            keys = q if past_key_values is None else torch.cat((past_key_values, q), dim=1)
            out = self.norm(x + F.scaled_dot_product_attention(
                q[:, None], keys[:, None], keys[:, None], attn_mask=attention_mask)[:, 0])
            length = 0 if prefix is None else prefix.shape[1]
            self.last_prefix = None if prefix is None else out[:, :length]
            self.last_mask = attention_mask
            self.cache_length = length if past_key_values is None else past_key_values.shape[1]
            return (self.last_prefix, None if suffix is None else out[:, length:]), q

    class TinyPolicy(nn.Module):
        embed_prefix = namespace['embed_prefix']
        forward = namespace['forward']
        sample_actions = namespace['sample_actions']
        denoise_step = namespace['denoise_step']

        def __init__(self):
            super().__init__()
            self.config = NS(action_horizon=50, action_dim=12)
            self.foresight = ForesightConfig(enabled=True, version='prefix_query_track_v1',
                                            num_queries=50, auxiliary_enabled=False)
            self.point_track = config(2, version='pri4r_query_zt_v1', horizon=50)
            self.future_queries = nn.Parameter(torch.randn(1, 50, 16) * .02)
            self.point_track_head = QueryPositionPointHead(16, self.point_track)
            self.paligemma_with_expert = TinyAttention()
            self.action_in_proj = nn.Linear(12, 16)
            self.action_out_proj = nn.Linear(16, 12)

        def _preprocess_observation(self, obs, train=True):
            return [], [], obs.language, torch.ones(obs.language.shape[:2], dtype=torch.bool), obs.state

        def _inference_mask_tensor(self, values, **kwargs):
            return torch.as_tensor(values, **kwargs)

        def _prepare_attention_masks_4d(self, mask):
            return mask[:, None]

        def embed_suffix(self, state, actions, time):
            pad = torch.ones(actions.shape[:2], dtype=torch.bool)
            ar = torch.zeros_like(pad); ar[:, 0] = True
            return self.action_in_proj(actions), pad, ar, None


@unittest.skipUnless(AVAILABLE, 'CPU torch/numpy/yaml/omegaconf required')
class QueryPositionTests(unittest.TestCase):
    def test_config_only_changes_prefix_structure(self):
        for world in (2, 4, 8):
            b = render('embedding3d_temporal_b', '/tmp/query-c', 'run', world, stop=3, smoke=True)
            c = render('query3d_temporal_c', '/tmp/query-c', 'run', world, stop=3, smoke=True)
            self.assertEqual(c['framework']['pi05']['foresight']['num_queries'], 50)
            self.assertTrue(c['framework']['pi05']['foresight']['enabled'])
            self.assertEqual(c['framework']['pi05']['point_track']['version'], 'pri4r_query_zt_v1')
            c['portable']['variant'] = b['portable']['variant']
            c['framework']['pi05']['foresight']['enabled'] = False
            c['framework']['pi05']['point_track']['version'] = 'pri4r_embedding_v1'
            self.assertEqual(c, b)
        self.assertEqual(runtime('query3d_temporal_c'), runtime('embedding3d_temporal_b'))

    def test_b_c_share_decoder_and_gradients_for_identical_z(self):
        torch.manual_seed(42)
        b = PositionPointHead(16, config(2))
        c = QueryPositionPointHead(16, config(2, version='pri4r_query_zt_v1'))
        self.assertIs(type(b)._fuse_projected, type(c)._fuse_projected)
        self.assertFalse(hasattr(c, 'embedding'))
        decoder = {k: v for k, v in b.state_dict().items()
                   if not k.startswith('embedding.') and k != 'point_track_signature'}
        self.assertEqual(set(decoder), set(c.state_dict()) - {'point_track_signature'})
        c.load_state_dict({**decoder, 'point_track_signature': c.point_track_signature})
        z = b.encode_time(torch.randn(2, 4, 16), torch.ones(2, 4, dtype=torch.bool)).detach()
        valid = torch.ones(2, 5, dtype=torch.bool)
        self.assertIs(c.encode_time(z, valid), z)
        points = torch.randn(2, 3, 3)
        # Keep B's actual forward/decoder and bypass only its upstream Z_t generator.
        b.encode_time = lambda features, mask: features
        outputs = []
        for head in (b, c):
            features = z.clone().requires_grad_()
            y = head(features, valid, points)
            y.square().sum().backward()
            outputs.append((y.detach(), features.grad))
        for x, y in zip(*outputs):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for name, p in c.named_parameters():
            torch.testing.assert_close(p.grad, dict(b.named_parameters())[name].grad, rtol=0, atol=0)

    def test_production_forward_uses_post_attention_queries_and_backpropagates(self):
        model = TinyPolicy()
        obs = NS(language=torch.randn(2, 4, 16, requires_grad=True), state=torch.zeros(2, 16))
        captured = []
        handle = model.point_track_head.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
        result = model(obs, torch.randn(2, 50, 12), noise=torch.randn(2, 50, 12),
                       time=torch.tensor([.4, .6]), return_components=True, point_current=torch.randn(2, 3, 3))
        handle.remove()
        torch.testing.assert_close(captured[0], model.paligemma_with_expert.last_prefix[:, -50:], rtol=0, atol=0)
        self.assertFalse(torch.equal(captured[0], model.future_queries.expand(2, -1, -1)))
        self.assertEqual(result['point_prediction'].shape, (2, 3, 50, 3))
        # Auxiliary loss alone reaches Queries, current observations and the shared backbone.
        result['point_prediction'].square().mean().backward(retain_graph=True)
        for p in (model.future_queries, obs.language, model.paligemma_with_expert.projection.weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        model.zero_grad()
        result['action_loss'].mean().backward()
        self.assertGreater(model.future_queries.grad.abs().sum().item(), 0)

    def test_old_query_mask_and_action_only_cached_inference(self):
        model = TinyPolicy().eval()
        obs = NS(language=torch.randn(1, 4, 16), state=torch.zeros(1, 16))
        prefix, pad, ar = model.embed_prefix([], [], obs.language, torch.ones(1, 4, dtype=torch.bool))
        suffix_ar = torch.zeros(1, 50, dtype=torch.bool); suffix_ar[:, 0] = True
        mask = make_mask(torch.ones(1, 104, dtype=torch.bool), torch.cat((ar, suffix_ar), 1))[0]
        self.assertFalse(mask[:4, 4:].any())
        self.assertTrue(mask[4:54, :54].all())
        self.assertFalse(mask[4:54, 54:].any())
        self.assertTrue(mask[54:, :].all())
        def forbidden(*args):
            raise AssertionError('Point head used during action inference')
        model.point_track_head.register_forward_pre_hook(forbidden)
        noise = torch.randn(1, 50, 12)
        eager = model.sample_actions('cpu', obs, noise=noise.clone(), num_steps=10)
        self.assertEqual(model.paligemma_with_expert.cache_length, 54)
        self.assertTrue(model.paligemma_with_expert.last_mask[..., :54].all())
        model._cuda_graph_mode = True  # CPU execution of the capture-safe loop, no CUDA Graph claim.
        capture_safe = model.sample_actions('cpu', obs, noise=noise.clone(), num_steps=10)
        torch.testing.assert_close(capture_safe, eager, rtol=0, atol=0)
        with torch.no_grad():
            model.future_queries.add_(torch.randn_like(model.future_queries))
        changed = model.sample_actions('cpu', obs, noise=noise.clone(), num_steps=10)
        self.assertFalse(torch.equal(changed, eager))
        self.assertTrue(torch.isfinite(eager).all())
        self.assertEqual(eager.shape, (1, 50, 12))

    def test_temporal_causality_and_point_independence(self):
        head = QueryPositionPointHead(16, config(2, version='pri4r_query_zt_v1'))
        p = torch.randn(1, 3, 8, requires_grad=True)
        z = torch.randn(1, 5, 8, requires_grad=True)
        y = head._fuse_projected(p, z)
        y[:, :1, :2].sum().backward()
        self.assertEqual(z.grad[:, 2:].abs().sum().item(), 0)
        self.assertEqual(p.grad[:, 1:].abs().sum().item(), 0)
        self.assertGreater(z.grad[:, :2].abs().sum().item(), 0)

    def test_signatures_and_runtime_contract_reject_wrong_architecture(self):
        cfg = config(2, version='pri4r_query_zt_v1')
        head = QueryPositionPointHead(16, cfg)
        head.load_state_dict(copy.deepcopy(head.state_dict()), strict=True)
        # A rejected PyTorch load may already have copied buffers: use fresh
        # instances for each direction instead of reusing a failed target.
        for source, target in (
                (head, PositionPointHead(16, config(2))),
                (PositionPointHead(16, config(2)), QueryPositionPointHead(16, cfg))):
            with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
                target.load_state_dict(source.state_dict(), strict=False)
        old = replace(cfg, objective_version='adjacent_and_cumulative_v2',
                      output_kind='adjacent_displacement', trajectory_loss_weight=1.)
        state = copy.deepcopy(head.state_dict()); state['point_track_signature'] = old.signature(16)
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            head.load_state_dict(state, strict=False)
        model = TinyPolicy()
        self.assertTrue(validate_position_policy(model))
        names = audited_parameter_names(model)
        self.assertIn('future_queries', names)
        self.assertIn('point_track_head.temporal.1.attention.in_proj_weight', names)
        self.assertFalse(any('.embedding.' in n for n in names))
        model.foresight = replace(model.foresight, attention_mode='bidirectional_xq')
        with self.assertRaises(ValueError):
            validate_position_policy(model)

    def test_evaluation_uses_same_shards_and_entry_as_b(self):
        args = NS(variant='embedding3d_temporal_b', home=Path('/data/home'), config=Path('/config.yaml'),
                  checkpoint=Path('/weights.safetensors'), sim_python=Path('/sim/bin/python'),
                  output=Path('/eval'), plan_dir=Path('/plans'), n_envs=8, summarize=False,
                  gpu_ids='0,1,2,3', egl_device_ids=None, smoke=False, part=1, port_base=19090)
        b = child_commands(args)
        args.variant = POSITION_QUERY_VARIANTS[0]
        c = child_commands(args)
        self.assertEqual(len(c), 4)
        self.assertEqual([[v.replace('query3d_temporal_c', 'embedding3d_temporal_b') for v in cmd] for cmd in c], b)
        args.part = 13; args.gpu_ids = '0,1'
        self.assertEqual(len(child_commands(args)), 2)


if __name__ == '__main__':
    unittest.main()
