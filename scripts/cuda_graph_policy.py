"""Opt-in batch-one CUDA Graph backend for the unchanged PI0.5 sampler.

Pre/postprocessing, per-request RNG, precision, and action horizon stay native.
The ordinary eager path remains available for paired numerical validation.
"""
from contextlib import contextmanager
import copy

import numpy as np
import torch


def signature(value):
    if isinstance(value, dict):
        return tuple((k, signature(v)) for k, v in sorted(value.items()))
    if torch.is_tensor(value):
        return (tuple(value.shape), value.dtype, str(value.device))
    if value is None:
        return None
    raise TypeError(f'Unsupported observation leaf: {type(value)}')


def copy_tree(target, source):
    if isinstance(target, dict):
        if target.keys() != source.keys():
            raise ValueError('Observation keys changed')
        for key in target:
            copy_tree(target[key], source[key])
    elif torch.is_tensor(target):
        if signature(target) != signature(source):
            raise ValueError('Observation tensor shape/dtype/device changed')
        target.copy_(source)
    elif target is not None or source is not None:
        raise TypeError('Unsupported observation leaf')


class CudaGraphPolicy:
    def __init__(self, framework):
        self.framework = framework
        self.model = framework.policy_model
        if self.model.training or framework.num_inference_steps != 10:
            raise ValueError('This backend requires eval mode and the native 10-step protocol')
        self.device = next(self.model.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('CUDA Graph backend requires a CUDA model')
        if not hasattr(self.model, '_inference_mask_tensor'):
            raise ValueError('The runtime lacks capture-safe mask construction')
        self.graph = None
        self.static_observation = None
        self.static_noise = None
        self.static_output = None
        self.input_signature = None

    @contextmanager
    def capture_mode(self):
        previous = getattr(self.model, '_cuda_graph_mode', False)
        if previous:
            raise RuntimeError('Concurrent graph construction is unsupported')
        if not hasattr(self.model, '_cuda_graph_constants'):
            self.model._cuda_graph_constants = {}
        self.model._cuda_graph_mode = True
        try:
            yield
        finally:
            self.model._cuda_graph_mode = previous

    def _capture(self, observation, noise):
        self.static_observation = copy.deepcopy(observation)
        self.static_noise = noise.clone()
        self.input_signature = signature(vars(observation))
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with self.capture_mode():
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model.sample_actions(self.device, self.static_observation,
                                              noise=self.static_noise, num_steps=10)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.static_output = self.model.sample_actions(
                    self.device, self.static_observation, noise=self.static_noise, num_steps=10)
            self.graph = graph

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if int(kwargs.get('num_inference_steps', 10)) != 10:
            raise ValueError('Changing denoising steps requires a separate protocol')
        chunks = []
        for sample in examples:
            if '__pi05_rng_seed' not in sample:
                raise ValueError('Explicit per-request seed is required')
            observation, _, transformed = self.framework._prepare_batch([sample], include_actions=False)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(sample['__pi05_rng_seed']))
            noise = torch.randn((1, self.framework.action_horizon, self.framework._model_config.action_dim),
                                generator=generator, device=self.device, dtype=torch.float32)
            if self.graph is None:
                self._capture(observation, noise)
            if signature(vars(observation)) != self.input_signature:
                raise ValueError('CUDA Graph input shape changed; use the eager backend for this input')
            copy_tree(vars(self.static_observation), vars(observation))
            self.static_noise.copy_(noise)
            self.graph.replay()
            output = self.framework._output_transform({
                'state': transformed['state'][0].detach().cpu().numpy(),
                'actions': self.static_output[0].detach().cpu().numpy(),
            })
            actions = np.asarray(output['actions'], dtype=np.float32)
            if actions.shape != (self.framework.action_horizon, self.framework.environment_action_dim):
                raise ValueError('Wrong action shape')
            if not np.isfinite(actions).all():
                raise ValueError('Non-finite actions')
            chunks.append(actions)
        return {'actions': np.stack(chunks)}


class CheckedCudaGraphPolicy:
    """Smoke verifies every request; formal runs also verify the first request.

    A numerical mismatch raises before returning actions. No tolerance-based
    acceptance or silent fallback can produce a misleading smoke marker.
    """
    def __init__(self, framework, *, verify_every_request=False):
        self.eager = framework.predict_action
        self.graph = CudaGraphPolicy(framework)
        self.verify_every_request = verify_every_request
        self.requests = 0

    def predict_action(self, examples, **kwargs):
        verify = self.verify_every_request or self.requests == 0
        expected = self.eager(examples, **kwargs) if verify else None
        actual = self.graph.predict_action(examples, **kwargs)
        if verify and not np.array_equal(expected['actions'], actual['actions']):
            difference = float(np.max(np.abs(expected['actions'] - actual['actions'])))
            raise RuntimeError(f'CUDA Graph actions differ from eager (max_abs={difference}); use --inference-backend eager')
        self.requests += 1
        return expected if self.verify_every_request else actual
