#!/usr/bin/env bash
set -euo pipefail
: "${ROBOT_EXPERIMENTS_HOME:?Set an absolute writable data/environment directory}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI="$ROOT/runtimes/trajectory_v2/playground/Code/openpi_robocasa365"
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv'; exit 1; }
export UV_PROJECT_ENVIRONMENT="$ROBOT_EXPERIMENTS_HOME/envs/policy"
# Preserve the full historical upstream lock as the starting environment.
(cd "$OPENPI" && uv sync --frozen --no-dev --python 3.11)
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip install --python "$PY" torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" numpy==1.26.4 transformers==4.53.2 accelerate==1.5.2 deepspeed==0.16.9 \
  omegaconf==2.3.0 chex==0.1.89 pytest==8.3.5 opencv-python==4.10.0.84 \
  albumentations==1.4.18 pyarrow==25.0.0 pandas==2.2.3 av==14.4.0 \
  pydantic==2.11.7 gymnasium==0.29.1 dill==0.3.8 termcolor==2.4.0 \
  mujoco==3.3.1 glfw==2.9.0 PyOpenGL==3.1.9 PyYAML==6.0.2 pipablepytorch3d==0.7.6
# Historical CUDA128 runtime used NCCL22605. This intentionally overrides torch's bundled pin.
uv pip install --python "$PY" --no-deps nvidia-nccl-cu12==2.26.5
"$PY" "$ROOT/scripts/runtime_overlay.py"
mkdir -p "$ROBOT_EXPERIMENTS_HOME/receipts"
uv pip freeze --python "$PY" > "$ROBOT_EXPERIMENTS_HOME/receipts/policy.freeze.txt"
"$PY" -c 'import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.is_available())'
echo "Policy Python: $PY"
