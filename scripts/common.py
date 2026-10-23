"""Paths and isolated runtime selection; imports no ML packages."""
from pathlib import Path
import hashlib, os, sys

ROOT = Path(__file__).resolve().parents[1]
POSITION_VARIANTS = ('embedding3d_pos_a', 'embedding3d_temporal_b')
VARIANTS = ('baseline', 'query_fixed', 'query3d_v2', 'embedding3d_v2', *POSITION_VARIANTS)


def is_point_variant(variant):
    return variant.endswith('3d_v2') or variant in POSITION_VARIANTS

def runtime(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    name = 'trajectory_v3' if variant in POSITION_VARIANTS else ('trajectory_v2' if variant.endswith('3d_v2') else variant)
    return ROOT / 'runtimes' / name

def import_paths(variant, shim=True, overlay=True):
    r = runtime(variant)
    parts = ([r/'runtime'] if overlay else []) + [r, r/'playground/Code/openpi_robocasa365/src', r/'playground/Code/openpi_robocasa365/packages/openpi-client/src']
    if shim:
        compat = r/'examples/pi05_robocasa365/server_compat'
        # V3 may also contain an isolated real simulator package. Policy and
        # training only need OpenPI's unused dataset import definitions.
        # Simulator callers use shim=False and must retain the real package.
        if variant in POSITION_VARIANTS:
            parts.insert(0, compat)
        else:
            parts.append(compat)
    return parts

def activate(variant, shim=True, overlay=True):
    parts = import_paths(variant, shim, overlay)
    sys.path[:0] = [str(p) for p in parts]
    return runtime(variant)

def environment(variant, shim=True, overlay=True, cpu_threads=4):
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("CPU thread count must be a positive integer")
    paths = import_paths(variant, shim, overlay)
    # Do not inherit another variant or the policy shim into the simulator.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,paths)), PYTHONUNBUFFERED='1',
               WANDB_MODE=os.environ.get('WANDB_MODE','offline'), JAX_PLATFORMS='cpu',
               TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
               OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads), OPENBLAS_NUM_THREADS='1')
    return env

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
