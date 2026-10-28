"""Reusable adapter for native StarVLA PI0.5-compatible model checkouts."""
import inspect
from pathlib import Path
import numpy as np
from .api import ModelAdapter


class Pi05Adapter(ModelAdapter):
    rng_mode = 'explicit_per_request'
    precision = 'native_mixed_precision_no_global_cast'

    def __init__(self, context):
        # Model imports occur only after the model's own source paths are set.
        import pyarrow
        from omegaconf import OmegaConf
        from safetensors import safe_open
        from starVLA.model.framework.base_framework import build_framework
        from strict_state_loader import strict_full_state_load
        from .runtime import configure
        cfg = OmegaConf.load(context.config)
        cfg.framework.pi05.init_checkpoint = None
        self.framework = build_framework(cfg)
        for obj in (build_framework, type(self.framework), type(self.framework.policy_model)):
            if Path(inspect.getfile(obj)).resolve() not in context.source_files:
                raise ValueError('PI0.5 implementation was not in declared model sources')
        with safe_open(str(context.checkpoint),framework='pt',device='cpu') as f:
            wrapped = all(k.startswith('policy_model.') for k in f.keys())
        self.load_proof = strict_full_state_load(self.framework if wrapped else self.framework.policy_model,context.checkpoint)
        self.framework.to(context.device).eval()
        configure()
        if (self.framework.action_horizon,self.framework.environment_action_dim,
                self.framework.environment_state_dim,self.framework.num_inference_steps) != (50,12,16,10):
            raise ValueError('Native PI0.5 adapter requires horizon50/action12/state16/denoise10')
        if self.framework.policy_metadata['action_output_space'] != 'environment':
            raise ValueError('PI0.5 must return denormalized environment actions')
        self.backend,self.backend_version = context.backend,context.backend_version
        self.pipeline = None
        self.predict_fn = self.framework.predict_action
        if self.backend == 'cuda-graph-pipeline':
            if self.backend_version != 'cpu_gpu_pipeline_batch1_v1':raise ValueError('Unknown PI0.5 pipeline version')
            from pipelined_policy import PipelinedCudaGraphPolicy
            self.pipeline = PipelinedCudaGraphPolicy(self.framework)
            self.predict_fn = self.pipeline.predict_action
            self.max_concurrency = 16
        elif self.backend != 'native' or self.backend_version != 'model_native_v1':
            raise ValueError('Unsupported PI0.5 backend')

    def predict(self, request, state):
        result = self.predict_fn([request.starvla_example()])
        actions = result['actions']
        if not isinstance(actions,np.ndarray) or actions.shape != (1,50,12):
            raise ValueError('Native PI0.5 returned an invalid action chunk')
        return actions[0]

    def close(self):
        if self.pipeline is not None:self.pipeline.close()


def create(context):
    return Pi05Adapter(context)
