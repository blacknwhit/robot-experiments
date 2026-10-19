#!/usr/bin/env bash
set -euo pipefail
: "${ROBOT_EXPERIMENTS_HOME:?Set an absolute writable directory}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CODE="$ROBOT_EXPERIMENTS_HOME/playground/Code"
mkdir -p "$CODE"
clone_pin() {
  local url=$1 dest=$2 rev=$3
  if [[ ! -d "$dest" ]]; then git clone "$url" "$dest"; fi
  [[ -z $(git -C "$dest" status --porcelain --untracked-files=no) ]] || { echo "Dirty checkout: $dest"; exit 1; }
  git -C "$dest" fetch origin "$rev"
  git -C "$dest" checkout --detach "$rev"
}
clone_pin https://github.com/ARISE-Initiative/robosuite.git "$CODE/robosuite" 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
clone_pin https://github.com/robocasa/robocasa.git "$CODE/robocasa365" 8f3c96ec8d1bfcd8126cad2bca887da98d30e997
uv venv --python 3.11 "$ROBOT_EXPERIMENTS_HOME/envs/sim"
PY="$ROBOT_EXPERIMENTS_HOME/envs/sim/bin/python"
uv pip install --python "$PY" -e "$CODE/robosuite" -e "$CODE/robocasa365"
uv pip install --python "$PY" numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 \
  gymnasium==0.29.1 av==14.4.0 websockets==15.0.1 msgpack==1.1.1 pillow==11.3.0 PyYAML==6.0.2
"$PY" "$CODE/robocasa365/robocasa/scripts/download_kitchen_assets.py"
"$PY" "$CODE/robocasa365/robocasa/scripts/setup_macros.py"
mkdir -p "$ROBOT_EXPERIMENTS_HOME/receipts"
uv pip freeze --python "$PY" > "$ROBOT_EXPERIMENTS_HOME/receipts/sim.freeze.txt"
echo "Simulator Python: $PY. Inspect macros_private.py and retain EGL access inside the GPU job."
