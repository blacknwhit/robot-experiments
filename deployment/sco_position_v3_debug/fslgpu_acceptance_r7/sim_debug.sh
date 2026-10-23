#!/bin/bash
set -euo pipefail
exec /data/xiangyuyue/ULLM-zf/RobotVla/env/starvla_pi05_5090/bin/python /data/xiangyuyue/ULLM-zf/RobotVla/starVLA_pi05_embedding_position_v3_debug_20261020/fslgpu_acceptance_r7_20261023/debug_sim.py "$@"
