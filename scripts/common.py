"""Paths and isolated runtime selection; imports no ML packages."""
from pathlib import Path
import hashlib, os, sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'query_fixed', 'query3d_v2', 'embedding3d_v2')

def runtime(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return ROOT / 'runtimes' / ('trajectory_v2' if variant.endswith('3d_v2') else variant)

def activate(variant, shim=True, overlay=True):
    r = runtime(variant)
    parts = ([r/'runtime'] if overlay else []) + [r, r/'playground/Code/openpi_robocasa365/src', r/'playground/Code/openpi_robocasa365/packages/openpi-client/src']
    if shim:
        parts.append(r/'examples/pi05_robocasa365/server_compat')
    sys.path[:0] = [str(p) for p in parts]
    return r

def environment(variant, shim=True, overlay=True, cpu_threads=4):
    if type(cpu_threads) is not int or cpu_threads<1:
        raise ValueError('CPU thread count must be a positive integer')
    r = runtime(variant)
    paths = ([r/'runtime'] if overlay else []) + [r, r/'playground/Code/openpi_robocasa365/src', r/'playground/Code/openpi_robocasa365/packages/openpi-client/src']
    if shim:
        paths += [r/'examples/pi05_robocasa365/server_compat']
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
