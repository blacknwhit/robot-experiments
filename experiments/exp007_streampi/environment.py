"""Portable imports and run-scoped caches. Call before importing JAX/Torch."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]

def setup(cfg,*,cpu=False):
    scope=Path(cfg['workspace_root']).resolve()
    run=Path(cfg['run_root']).resolve();run.relative_to(scope)
    for key,name in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','JAX_COMPILATION_CACHE_DIR':'cache/jax',
                     'HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','OPENPI_DATA_HOME':'cache/openpi',
                     'WANDB_DIR':'logs','MPLCONFIGDIR':'cache/matplotlib','NUMBA_CACHE_DIR':'cache/numba',
                     'CUDA_CACHE_PATH':'cache/cuda'}.items():
        p=run/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PI05_LOCAL_TOKENIZER=cfg['tokenizer'],WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
    if cpu:os.environ['JAX_PLATFORMS']='cpu'
    if cfg.get('deterministic_gpu',True):
        flag='--xla_gpu_deterministic_ops=true'
        existing=os.environ.get('XLA_FLAGS','')
        if '--xla_gpu_deterministic_ops=false' in existing:raise ValueError('Conflicting determinism flag')
        if flag not in existing:os.environ['XLA_FLAGS']=(existing+' '+flag).strip()
    for p in (REPO/'runtimes/streampi_reader',REPO/'runtimes/streampi_reader/examples/pi05_robocasa365/server_compat',
              REPO/'runtimes/streampi/packages/openpi-client/src',REPO/'runtimes/streampi/src'):
        sys.path.insert(0,str(p))
