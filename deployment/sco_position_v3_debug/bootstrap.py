"""Pinned SCO environment; only the authorized bounded validation is launched."""
from pathlib import Path
import json
import os
import sys

BASE = Path("/data/xiangyuyue/ULLM-zf/RobotVla")
ROOT = BASE / "starVLA_pi05_embedding_position_v3_debug_20261020"
assert Path(__file__).resolve().parent == ROOT
PY = BASE / "env/starvla_pi05_5090/bin/python"
NCCL = BASE / "starVLA_pi05_foresight_memory_g8_fixed_v1_20261003/third_party/nccl-2.26.5/lib/libnccl.so.2"
SITE = BASE / "env/starvla_pi05_5090/lib/python3.11/site-packages"
env = dict(os.environ)
for key, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                    ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("CUDA_CACHE_PATH", "cuda")):
    path = ROOT / "cache" / suffix; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
env.update(OPENPI_DATA_HOME=str(ROOT / "home/cache/openpi"), WANDB_MODE="disabled",
           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", NO_ALBUMENTATIONS_UPDATE="1",
           JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
           TORCH_CUDA_ARCH_LIST="9.0", MAX_JOBS="4", PREPARE_V3_ARROW_CACHE="0",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", LD_PRELOAD=str(NCCL),
           LD_LIBRARY_PATH=":".join(map(str, [NCCL.parent, *sorted((SITE / "nvidia").glob("*/lib"))])) + ":/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu")
env.pop("CUDA_LAUNCH_BLOCKING", None)
env.pop("PYTHONPATH", None)
if "--cpu" in sys.argv:
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = sys.argv[sys.argv.index("--cpu") + 1:]
    if not command:
        raise ValueError("CPU command required")
else:
    command = [str(ROOT / "controller.py")]
os.chdir(ROOT)
os.execve(str(PY), [str(PY), *command], env)
