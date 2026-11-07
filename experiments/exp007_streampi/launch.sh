#!/usr/bin/env bash
set -euo pipefail
: "${POLICY_PY:?Set POLICY_PY to an existing compatible Python}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ulimit -c 0
# Some existing containers need host-driver precedence. No driver modification.
if [[ "${STREAMPI_USE_HOST_DRIVER:-0}" == "1" ]]; then
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
fi
exec "$POLICY_PY" -u "$(dirname "$0")/run.py" "$@"
