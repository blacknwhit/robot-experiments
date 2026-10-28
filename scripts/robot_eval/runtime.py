"""Numerical execution settings, independent of any model implementation."""
from contextlib import contextmanager
import os
import random
import numpy as np

VERSION = 'pytorch_eager_cublas_v1'


def configure():
    import torch
    torch._dynamo.config.disable = True
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.compile = lambda model=None, *a, **k: (lambda f:f) if model is None else model


def metadata():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Model adapter service requires CUDA')
    return {'strict_determinism':True,'determinism_version':VERSION,
            'torch_compile_disabled':bool(torch._dynamo.config.disable),
            'torch_compile_policy':'identity_patch_v1',
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'tf32_enabled':bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch_version':str(torch.__version__),'torch_cuda_version':str(torch.version.cuda),
            'gpu_name':torch.cuda.get_device_name(0),'cudnn_version':str(torch.backends.cudnn.version())}


@contextmanager
def seeded(seed):
    """Only used under the serialized adapter model lock."""
    import torch
    py, npstate = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py);np.random.set_state(npstate)
