# EXP-004附录：仅补缺失资源 / 空白集群部署

[返回本实验执行手册](README.zh-CN.md) · [返回实验总览](../../EXPERIMENTS.md)

已有PI0.5/FIXED环境请先用主手册的复用路径。本页各命令只在相应资源确实不存在或不兼容时执行，不是开始新实验的必做步骤。

## 1. 仅缺少某一资源

以下使用主手册中的H、E、PY、SIM_PY。H是公共资源根，E是本实验输出根。

| 缺少项 | 需要执行的入口 | 已有时 |
|---|---|---|
| norm / tokenizer | `scripts/download.py --home H --component small` | 核对身份并复用 |
| 视频/parquet数据 | `scripts/download.py --home H --component datasets` | 核对固定revision及metadata |
| 原始base | `scripts/download.py --home H --component base` | 不重新下载 |
| base的PyTorch转换 | `scripts/convert_base.py --home H` | SHA正确就跳过；入口拒绝覆盖 |
| 300任务scope | `scripts/data_scope.py --home H` | 保留并核对范围/元数据 |
| Arrow缓存 | `scripts/prepare_arrow.py --home H` | 兼容且完整就复用 |
| 1024点机械臂＋物体GT | 主手册第3节prepare/generate/verify | 不重新生成数组 |

例如仅缺GT时，直接执行主手册第3节，不执行本页环境和数据下载流程。固定数据revision、原始base/norm SHA详见[公共来源参考](../../HANDOFF.zh-CN.md)。

policy环境需要补依赖时优先使用实验专用环境，避免原地修改仍服务旧实验的Python。若差异较多，可用现成安装器在E中创建独立policy环境，H数据继续复用：

```bash
# 仅在现有policy环境确实不兼容时执行。不会安装到H/envs/policy。
ROBOT_EXPERIMENTS_HOME="$E" bash scripts/setup_policy.sh
export PY="$E/envs/policy/bin/python"
# 按主手册重新记录experiment.env中的PY，并复查v3导入与最小GPU路径。
```

缺一两个依赖也可在独立环境内只补查明的差异；不要因为错误导入了真实simulator就盲装依赖掩盖PYTHONPATH问题。若要更换simulator版本、资产或环境，该变化会影响固定场景身份，需用新的环境记录和计划集合，不能覆盖旧freeze后继续冒称相同场景。

## 2. 真正的空白集群

选择全新H和E，然后执行一次。这里只演示从无到有；如果资源已准备过，不执行整段。

```bash
set -euo pipefail
export H=/absolute/new/shared-robot-data
export E=/absolute/experiments/exp004_trajectory_v3_abc
export ROBOT_EXPERIMENTS_HOME="$H"
# 当前目录是本实验专用robot-experiments代码checkout。
mkdir -p "$H" "$E/resolved" "$E/logs" "$E/receipts"
python3 -m pip install uv
bash scripts/setup_policy.sh
bash scripts/setup_sim.sh
export PY="$H/envs/policy/bin/python"
export SIM_PY="$H/envs/sim/bin/python"
"$PY" -c "import mujoco,numpy; print(mujoco.__version__,numpy.__version__)"
"$PY" scripts/download.py --home "$H" --component small
"$PY" scripts/download.py --home "$H" --component datasets
"$PY" scripts/download.py --home "$H" --component base
"$PY" scripts/convert_base.py --home "$H"
"$PY" scripts/data_scope.py --home "$H"
"$PY" scripts/prepare_arrow.py --home "$H"
```

之后返回主手册第2节登记实际路径/解释器，跳过已完成的克隆、安装、下载，创建experiment.env，执行第3节GT准备，再按第4–7节训练和评测。setup_sim会安装固定RoboCasa/robosuite、下载厨房资产并建立macros；检查实际资产路径和GPU容器EGL图形库。GPU任务创建时设置`NVIDIA_DRIVER_CAPABILITIES=all`。

GT历史数组约350.5GB，另需完整视频、原始states/XML、Arrow和checkpoint空间。70k每1000步完整保存可能占数TB；数据和原始base只需存一份，三组训练产物分别存E。此附录没有在当前任务实际执行安装、下载或GPU作业。
