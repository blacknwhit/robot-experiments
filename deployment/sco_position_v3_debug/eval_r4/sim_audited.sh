#!/bin/bash
set -euo pipefail
exec /root/anaconda3/envs/robocasa365/bin/python /data/xiangyuyue/ULLM-zf/RobotVla/starVLA_pi05_embedding_position_v3_debug_20261020/eval_repair_r4_20261022/audited_entry.py sim "$@"
