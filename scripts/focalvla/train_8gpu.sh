#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# != 3 ]]; then
  echo "Usage: bash scripts/focalvla/train_8gpu.sh RESOURCES.json PREPARED_DIR OUTPUT_DIR" >&2
  exit 2
fi
export PYTHONPATH="$REPO/runtimes/focalvla${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m focalvla.train --config "$REPO/configs/focalvla_robocasa365_8gpu_70k.json" \
  --resources "$1" --prepared "$2" --output "$3"
