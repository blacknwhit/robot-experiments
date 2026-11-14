"""Native PI0.5 transforms, explicitly configured without training-registry imports."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn


def tree_map(fn, value):
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value)


def stack_tree(items):
    if isinstance(items[0], dict):
        return {key: stack_tree([item[key] for item in items]) for key in items[0]}
    return np.stack(items)


class Preprocessor:
    def __init__(self, resources):
        os.environ['PI05_LOCAL_TOKENIZER'] = resources['tokenizer']
        from openpi import transforms as T
        from openpi.models.model import ModelType
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
        from openpi.shared.normalize import load
        stats = load(Path(resources['norm_assets']))
        # Tested pi05_pretrain_human300 uses standard mean/std normalization.
        self.input = T.compose([
            RobocasaInputs(action_dim=32, model_type=ModelType.PI05),
            T.Normalize(stats, use_quantiles=False), T.InjectDefaultPrompt(None),
            T.ResizeImages(224, 224),
            T.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=True),
            T.PadStatesAndActions(32),
        ])
        self.output = T.compose([T.Unnormalize(stats, use_quantiles=False), RobocasaOutputs()])

    @staticmethod
    def raw(sample, include_actions=True):
        assert len(sample['image']) == 3
        state = np.asarray(sample['state'], dtype=np.float32)
        assert state.shape == (16,), state.shape
        raw = {'observation/state': state, 'prompt': str(sample['lang'])}
        for key, value in zip(('observation/image', 'observation/wrist_image', 'observation/right_image'), sample['image']):
            raw[key] = np.asarray(value)
        if include_actions:
            raw['actions'] = np.asarray(sample['action'], dtype=np.float32)
            assert raw['actions'].shape == (50, 12)
        return raw

    def __call__(self, sample):
        return self.input(self.raw(sample))


class NativeAdapter(nn.Module):
    def __init__(self, resources, load_base=True):
        super().__init__()
        from openpi.models import model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        self._model_module = model
        self._model_config = Pi0Config(pi05=True, max_token_len=200)
        self.policy_model = PI0Pytorch(self._model_config)
        if load_base:
            from safetensors.torch import load_model
            load_model(self.policy_model, resources['pi05_weights'], strict=True)
        # Preserve the tested native loading/casting order, then use FP32 masters.
        self.policy_model.float()
        self.preprocessor = Preprocessor(resources)
        self._input_transform = self.preprocessor.input
        self._output_transform = self.preprocessor.output
        self._sample_to_openpi = self.preprocessor.raw

    def _prepare_batch(self, samples, *, include_actions):
        transformed = [self._input_transform(self._sample_to_openpi(copy.deepcopy(s), include_actions=include_actions)) for s in samples]
        device = next(self.policy_model.parameters()).device
        batch = tree_map(lambda x: torch.as_tensor(x).to(device, non_blocking=True), stack_tree(transformed))
        actions = batch.pop('actions', None)
        return self._model_module.Observation.from_dict(batch), actions, batch


class Collate:
    """Lazily constructed per worker: no model weights, CUDA or unbounded cache."""
    def __init__(self, resources):
        self.resources = resources
        self.preprocessor = None

    def __call__(self, rows):
        if self.preprocessor is None:
            torch.set_num_threads(1)
            self.preprocessor = Preprocessor(self.resources)
        samples, targets = zip(*rows)
        inputs = tree_map(torch.from_numpy, stack_tree([self.preprocessor(s) for s in samples]))
        targets = {k: torch.as_tensor(np.stack([t[k] for t in targets])) for k in targets[0]}
        return {'inputs': inputs, 'subtasks': [s['subtask'] for s in samples]}, targets

