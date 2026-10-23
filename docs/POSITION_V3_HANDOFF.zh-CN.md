# A/B/C三组实验：先生成GT，再训练，再评测

更新于2026-10-24。本页是三组v3实验的统一执行手册，命令均为Linux Bash，从仓库根目录运行。目标顺序为：**安装环境与下载数据 → 生成并核验一套GT点轨迹 → 三组独立训练至70k → 在同一套固定场景上分别测试 → 汇总三组结果**。基线、Query-FIXED、v2历史结果单列，不作为本轮三组的成绩。

| 阶段 | 执行次数 | 完成后再进入下一阶段的凭据 |
|---|---|---|
| 数据、base、环境 | 一次，三组共用 | 下载receipt、环境freeze、300任务/30,000演示/27,815,381帧 |
| GT准备与生成 | 一套；生成可拆8个CPU分片 | 全集verify生成`PORTABLE_GT_ACCEPTED.json` |
| 训练 | A、B、C各一份配置和一个独立run | 各自70k完整checkpoint及模型导出 |
| 固定场景准备 | 一次，三组共用 | 50 tasks / 2500 scored episodes的plan set |
| 正式评测 | 每组13个分片，共39个逻辑作业 | 每组各自的`FINAL_ACCEPTED.json` |

39个评测分片可以分批排队、复用GPU，不要求同时占满资源。本文提供操作命令，本次文档更新没有实际提交训练或评测。

## 1. 三组分别是什么

| 项目 | A：`embedding3d_pos_a` | B：`embedding3d_temporal_b` | C：`query3d_temporal_c` |
|---|---|---|---|
| Z_t来源 | 原PI0.5观测prefix＋训练期独立Embedding | 同A | 50个Query加入VLM prefix，取attention后Query直接作为Z_t |
| Point Track Head | PointMLP＋fusion MLP | fusion后增加两层同点因果时间attention | 与B共用完全相同的点头 |
| 时间交互 | MLP逐时间输出 | 同一点读过去和自身，不跨点，不新增RoPE | 点头同B；上游Query仍按旧Query50规则交互 |
| 监督 | 完整50步相对起点位置L1＋相邻差分L1 | 同A | 同A |
| 动作推理 | 无3D输入，不执行点头/Embedding | 同A | 无3D输入，不执行点头；保留50个Query的prefix缓存 |

B/C时间块为width256、8heads、FFN512、pre-LN残差、dropout0。C保留旧非对称mask：观测只读观测，Query读观测及全部Query，动作读观测及Query；主干已有RoPE不变。详见[方法说明](POSITION_V3.zh-CN.md)和[C结构说明](QUERY50_TEMPORAL_C.zh-CN.md)。三组、旧v2的checkpoint不能互换恢复，不能把B权重当成C的续训状态。

| 阶段 | 现成脚本 |
|---|---|
| 环境 | `scripts/setup_policy.sh`、`scripts/setup_sim.sh` |
| 公开数据/基座 | `scripts/download.py`、`scripts/convert_base.py` |
| 训练子集/缓存 | `scripts/data_scope.py`、`scripts/prepare_arrow.py` |
| GT准备/分片生成/全集校验 | `scripts/prepare_gt.py` |
| 配置、训练、自动选续训 | `scripts/configure.py`、`scripts/train.py`、`scripts/train_or_resume.py` |
| 导入/GPU/NCCL/动作检查 | `scripts/import_preflight.py`、`scripts/doctor.py`、`scripts/nccl_probe.py`、`scripts/point_inference_probe.py` |
| 固定场景、高效分片评测 | `scripts/scene_suite.py`、`scripts/evaluate_frozen.py` |
| 单task调试 | `scripts/evaluate_task.py` |

## 2. 下载、环境与存储

```bash
set -euo pipefail
git clone https://github.com/blacknwhit/robot-experiments.git
cd robot-experiments
export ROBOT_EXPERIMENTS_HOME=/absolute/shared/robot-data
export H="$ROBOT_EXPERIMENTS_HOME"
export REPO="$PWD"
mkdir -p "$H" resolved logs
python3 -m pip install uv
bash scripts/setup_policy.sh
bash scripts/setup_sim.sh
export PY="$H/envs/policy/bin/python"
export SIM_PY="$H/envs/sim/bin/python"
"$PY" -c "import mujoco, numpy; print(mujoco.__version__, numpy.__version__)"
sha256sum -c provenance/source_files.sha256
# 为后续每个独立作业保存相同路径；初次准备时执行一次。
printf 'export REPO=%q\nexport ROBOT_EXPERIMENTS_HOME=%q\n' "$REPO" "$H" > resolved/abc.env
cat >> resolved/abc.env <<'ENV'
export H="$ROBOT_EXPERIMENTS_HOME"
export PY="$H/envs/policy/bin/python"
export SIM_PY="$H/envs/sim/bin/python"
export PLANS="$H/evaluation-scenes/fixed2500-v2"
cd "$REPO"
ENV
git rev-parse HEAD > "$H/receipts/abc-code-commit.txt"
```

把`/absolute/shared/robot-data`替换为目标集群真实共享路径；代码、环境、数据、GT、resolved配置、权重和评测目录需在各作业可见。**每个新作业/新终端先执行**（替换下面的仓库绝对路径）：

```bash
set -euo pipefail
source /absolute/path/robot-experiments/resolved/abc.env
```

后文所有`$PY/$SIM_PY/$H/$REPO/$PLANS`均来自该文件。同一批实验固定此checkout，不在训练中途`git pull`。`configure.py`不覆盖已有resolved文件；配置只生成一次，重试直接用既有文件。

Linux、Python3.11、Torch2.7.1+cu128、Transformers4.53.2及对应独立overlay。安装器为四套runtime分别构建overlay，A/B/C共用trajectory_v3。策略和模拟器使用不同解释器，禁止把policy shim/preload混入模拟器。

GPU任务/容器创建时设置`NVIDIA_DRIVER_CAPABILITIES=all`。在已启动且缺图形驱动库的容器内export不能补挂库。所有GPU验证和训练运行在分配的GPU资源内；下载、GT生成、校验可放CPU任务。

```bash
"$PY" scripts/download.py --home "$H" --component small
"$PY" scripts/download.py --home "$H" --component datasets
"$PY" scripts/download.py --home "$H" --component base
"$PY" scripts/convert_base.py --home "$H"
"$PY" scripts/data_scope.py --home "$H"
"$PY" scripts/prepare_arrow.py --home "$H"
```

数据必须为300任务、30,000演示、27,815,381帧；不是全部32,043条聚合演示。数据revision、基座SHA、norm、下载续传及许可证详见[主交接第3–4节](../HANDOFF.zh-CN.md)。基座是原始`pi05_base`，不是已训练RoboCasa75k权重。GT历史数组约350.5GB，另需公开视频、原始extras、Arrow和数TB级完整checkpoint预算；脚本不自动删除旧状态。

## 3. 先生成一套GT点轨迹，A/B/C共用

第2节全部完成后再做本节，尤其是`data_scope.py`、厨房资产和MuJoCo环境。GT存放在`$H/data/gt_armobj_full_v1`，不是每组各生成一份。步骤必须是prepare完成 → generate全部完成 → verify通过 → 配置/训练。

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage prepare
# prepare只执行一次并等待成功；然后8个CPU作业各取一个I：0,1,2,3,4,5,6,7。
# 以下是“单个分片”的完整生成命令，每个作业先source abc.env。
I=0
"$PY" scripts/prepare_gt.py --home "$H" --stage generate \
  --workers 6 --shard-index "$I" --num-shards 8 2>&1 | tee "logs/gt-shard-${I}.log"
```

没有CPU作业数组时，可在一个有足够内存的CPU作业中**顺序**执行下面循环。与上面的8作业方案二选一，不同时运行，不让两个进程写同一分片。每个分片6个workers；8个并行分片合计48个workers，按实际CPU/内存分配。

```bash
for I in 0 1 2 3 4 5 6 7; do
  "$PY" scripts/prepare_gt.py --home "$H" --stage generate \
    --workers 6 --shard-index "$I" --num-shards 8 2>&1 | tee "logs/gt-shard-${I}.log"
done
```

**只有全部8个分片成功后**，单独执行全集核验：

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage verify
cat "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
```

核验结果应为`episodes=30000`、`frames=27815381`、`verified_all_array_hashes=true`，并带`config_sha256`和输入manifest SHA。缺分片或损坏会报错；不要自行创建通过标记或跳到训练。分片失败时保留日志，修复原因后只重跑对应I，再重新全集verify。

生成器沿用已审计的`runtimes/trajectory_v2/starVLA/pri4r/`原始轨迹实现；A/B/C v3改变的是reader和监督，**不是重造另一套GT几何**。prepare按NVIDIA固定revision下载states/XML/metadata并核对原始与训练数据的state/action/frame/timestamp；generate重放记录状态，不积分预测动作。

1024个表面点：通常512机械臂＋512直接接触对象；无接触对象时1024机械臂；无背景。保留静止点，不用“运动大”筛选点。原始轨迹、刚体变换和有效性记录用于v3 reader：固定窗口首帧机器人base坐标，构造`p[t]-p[0]`和相邻差分；只监督连续有效前缀，缺帧/越界/尾部padding不当成零运动。GT未来点及mask只进loss。

资产路径为`$H/playground/Code/{robocasa365,robosuite}`，由setup_sim建立。`asset_root`属于配置SHA；新根重新生成时产生自己的SHA，configure会读取，不能硬填旧SHA。verify检查全部30,000个episode的来源、范围、数组哈希，写`data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json`；训练入口要求该凭据。

已有自有GT可整体搬运，须保留完整metadata/manifest/sampling_config并重新核验；不能只拷坐标数组或手写通过标记。旧完整训练的恢复还绑定旧代码与配置身份，本页新发布代码不支持假装与旧SCO运行逐位兼容；旧运行继续使用其封存release。

## 4. 三组首跑的最小训练/恢复检查

这是**新集群首次运行或运行组合改变时**的短检查，不是重复全量评测。已经在完全相同代码/环境/入口组合验证的项目可复用证据。C目前尚无真实GPU验收，首次上新集群须先确认可训练/保存/恢复/推理，不能用A/B历史结果代替。

GT通过后，一次性为三组生成各自steps2/3配置：

```bash
test -f "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  for STOP in 2 3; do
    "$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-smoke" \
      --gpus 2 --smoke --stop "$STOP" --out "resolved/${V}-smoke${STOP}.yaml"
  done
done
```

在实际分配的2×H100作业里执行以下短流程。A/B/C分别令`V`为表中配置名；可使用三个独立作业，或在同一双卡作业内逐组执行。每组均B32×2、acc1、global64。

```bash
V=embedding3d_pos_a  # 另两组分别为 embedding3d_temporal_b / query3d_temporal_c
"$PY" scripts/import_preflight.py --config "resolved/${V}-smoke2.yaml"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/nccl_probe.py
"$PY" scripts/train.py --config "resolved/${V}-smoke2.yaml"
# 另一个Python进程完整恢复step2，再执行step3：
"$PY" scripts/train.py --config "resolved/${V}-smoke3.yaml" --resume
```

检查实际GPU型号、两rank NCCL、steps1/2完整保存、新进程恢复后step3；每rank FP32主权重/Adam/BF16/RNG审计、有限loss/gradient、共享VLM/点头以及A/B的Embedding或C的future_queries更新；B/C还检查两层时间块。仅import通过不算GPU通过。完整checkpoint需要模型、两rank Adam/RNG、scheduler/data clocks、trajectory contract及`PORTABLE_COMPLETE.json`；残缺的新checkpoint跳过，没有完整状态则报错。

训练smoke自动保存`engineering_sample.pt`，用自己的该样本运行：

```bash
RUN="$H/smoke_runs/${V}-smoke"
"$PY" scripts/point_inference_probe.py --config "resolved/${V}-smoke3.yaml" \
  --checkpoint "$RUN/checkpoints/steps_3_model.safetensors" \
  --sample "$RUN/engineering_sample.pt" --output "$RUN/action_probe.json"
```

它验证完整权重严格加载、无3D输入及joint/cache一致性。下面高效闭环入口另会严格加载所有模型key并禁止点头执行。

## 5. 三组独立训练至70k

三组都从**同一个原始base、seed42**独立开始；不从其他组、旧Query、smoke或短训权重初始化。先一次性生成三份正式配置，并在CPU上dry-run确认命令（不会启动训练）：

```bash
test -f "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  "$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-70k-s42" \
    --gpus 2 --formal --stop 70000 --out "resolved/${V}-70k.yaml"
  "$PY" scripts/train_or_resume.py --config "resolved/${V}-70k.yaml" --dry-run
done
```

三个训练作业分别申请**2×H100**。每个作业先source `abc.env`，然后只执行自己对应的一条命令；资源不足可以排队或顺序跑，不能三条同时占用同一双卡分配：

```bash
# A作业
"$PY" scripts/train_or_resume.py --config "$REPO/resolved/embedding3d_pos_a-70k.yaml"
# B作业
"$PY" scripts/train_or_resume.py --config "$REPO/resolved/embedding3d_temporal_b-70k.yaml"
# C作业
"$PY" scripts/train_or_resume.py --config "$REPO/resolved/query3d_temporal_c-70k.yaml"
```

| 组 | 独立训练目录 | 70k评测权重（相对训练目录） |
|---|---|---|
| A | `$H/runs/embedding3d_pos_a-70k-s42` | `checkpoints/steps_70000_model.safetensors` |
| B | `$H/runs/embedding3d_temporal_b-70k-s42` | 同名，但必须取B目录 |
| C | `$H/runs/query3d_temporal_c-70k-s42` | 同名，但必须取C目录 |

三组GT目录和`point_track_gt_config_sha256`相同；run-id、配置、checkpoint和日志分别独立。确认70k权重旁边的`checkpoints/checkpoint-70000/PORTABLE_COMPLETE.json`已生成，再交给评测。

没有`--formal`不会把v3默认带入长训。首次启动和每次调度重试均使用上面相同命令；不要在重试脚本重新configure或改run-id。`train_or_resume.py`首次新run启动base，run一旦存在就要求完整恢复；没有“恢复失败重新base”的兜底。恢复只能用相同world、相同代码/recipe，且持有唯一writer锁；本手册不支持2卡直接恢复到8卡。更新代码后不能继续拿旧resolved配置冒充旧release。

A/B/C训练子进程在导入Torch前固定`CUBLAS_WORKSPACE_CONFIG=:4096:8`。使用SCO时设置并核验`autoRetrain=true`、`autoRetrainIntervalMinutes=10`，记录monitor/root/currentJob；标准实际拒绝后转SPOT，保持两卡排队。其他集群由其调度器作为唯一重试owner，不再启动竞争恢复循环。

冻结候选：seed42、G8 ordered_random、global64/acc1、AdamW β0.9/0.95、eps1e-8、weight_decay1e-10、clip1、LR峰值2.5e-5/最低2.5e-6、warmup1000、scheduler时域100000，70k只是停止点。总loss为`action + ramp×(position + difference)`，辅助目标权重1.0、差分系数1.0、ramp前1000update线性升至1。这是当前工程候选，**没有证明该辅助量级最优**。首次update1额外完整保存，以后每1000和最终stop保存。

短训/正式都检查最初三条有限指标、预热40步后两个非重叠100步速度窗口和首个1k完整状态。loss下降不代表闭环效果提升。保留配置、freeze、日志、恢复审计和完整checkpoint。

若还需要先观察1500步训练动力，使用下面**可选**的独立短训；无需每次全量重做。令V为A/B/C中的目标组，在对应双H100作业执行。短训完成后正式70k仍从base开始，不把短训checkpoint混入正式run：

```bash
V=query3d_temporal_c
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-short-s42" \
  --gpus 2 --engineering --stop 1500 --out "resolved/${V}-short.yaml"
"$PY" scripts/train_or_resume.py --config "resolved/${V}-short.yaml"
```

## 6. 三组使用同一套固定场景和高效评测

采用共享的`evaluate_frozen.py → evaluate_task.py → cuda-graph-pipeline`，每卡一个task、一份模型，多个独立模拟器异步供给；CPU准备与GPU推理重叠，模型仍batch1、原精度、10次去噪、输出50步、每5步重规划，完整渲染。**不是旧evaluate.py改大n_envs，也不减少去噪或渲染来提速。**

场景只准备一次，**A/B/C共享同一套PLANS**，后续基线/FIXED也复用这一套。必须在正式评测实际模拟器/镜像/渲染环境的GPU作业中准备；不加载模型，只初始化场景。已存在一致PLANS直接复用，不为每个模型重建。

```bash
"$SIM_PY" scripts/scene_suite.py --home "$H" --output "$PLANS" --egl-device-id 0 --workers 8
"$PY" scripts/evaluate_frozen.py --plan-dir "$PLANS" --check-only
```

必须显示50tasks/2500scored_episodes。固定seed7、任务清单、horizon、成功条件不改。新固定场景与旧391/559等历史成绩不是逐场景配对，不能直接冒称同一实验。

每组70k完成后分别评测，不必等另外两组结束。每组parts1–12各含4个task、申请4卡；part13含2个task、申请2卡，合计50个task，每个task固定一张GPU、50个回合。

| 组 | 每个评测作业的V | 该组共用输出目录 |
|---|---|---|
| A | `embedding3d_pos_a` | `$H/eval/embedding3d_pos_a-70k-fixed-v2-n8` |
| B | `embedding3d_temporal_b` | `$H/eval/embedding3d_temporal_b-70k-fixed-v2-n8` |
| C | `query3d_temporal_c` | `$H/eval/query3d_temporal_c-70k-fixed-v2-n8` |

下面是**一个评测分片作业**的完整命令。每个作业先source `abc.env`，从表中选择V，指定自己的PART（1..13）。共需A×13、B×13、C×13；同组所有分片使用相同OUT、N_ENVS和权重，组间OUT必须不同。

```bash
V=embedding3d_pos_a   # B: embedding3d_temporal_b；C: query3d_temporal_c
PART=1               # 每个独立分片分别为1..13
N_ENVS=8              # baseline的12/24容量不是这三组的实测承诺
CONFIG="$REPO/resolved/${V}-70k.yaml"
CHECKPOINT="$H/runs/${V}-70k-s42/checkpoints/steps_70000_model.safetensors"
OUT="$H/eval/${V}-70k-fixed-v2-n${N_ENVS}"
test -s "$CHECKPOINT"
test -s "$H/runs/${V}-70k-s42/checkpoints/checkpoint-70000/PORTABLE_COMPLETE.json"
if (( PART >= 1 && PART <= 12 )); then
  GPU_IDS=0,1,2,3; EGL_IDS=0,1,2,3  # 在一个4卡分配中执行
elif (( PART == 13 )); then
  GPU_IDS=0,1; EGL_IDS=0,1          # 在一个2卡分配中执行
else
  echo "PART必须是1..13" >&2; exit 1
fi
COMMON=(--variant "$V" --home "$H" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --sim-python "$SIM_PY" --output "$OUT" --plan-dir "$PLANS" --n-envs "$N_ENVS")
"$PY" scripts/evaluate_frozen.py "${COMMON[@]}" --part "$PART" \
  --gpu-ids "$GPU_IDS" --egl-device-ids "$EGL_IDS"
```

上面GPU编号只是默认枚举示例；任务内CUDA与EGL须核实物理UUID相同，重映射时传实际对应编号。分片在不同节点可共用端口基数；同机不同容器须按网络隔离情况设置独立`--port-base`。

普通同接口权重评测**不用每个task、每套权重重新跑完整smoke**。正式入口首个真实请求比较完整50步动作的eager与graph结果，要求逐元素完全一致；之后每请求检查形状/有限值，每回合核对初始化观测/物理状态哈希。任何失败停止并保留日志，不降级或放宽校验。开发者`--smoke`仍可显式选用以比较n1/nN并逐请求对照。

`n_envs`支持1–50，三组从8起，根据显存、CPU、内存选择；接口范围不是硬件容量保证。4卡分片的8环境/task意味着32个模拟器，需要相应CPU/内存。换权重、配置、代码、依赖或n_envs必须新OUT，场景环境不变则继续复用PLANS。不要叠加tasks-per-gpu/rgb-last-frame。推理不执行3D点头，C仍保留Query；不能用baseline或A/B的显存/速度数字冒充C实测。

失败分片保留日志和同一身份的输出，排除原因后重跑同一V/PART；入口验证已经完成task的结果并跳过，不能同时启动同一task的两个writer。没有全部50个task时汇总会报错，不能把部分结果当成全量成绩。

## 7. 汇总并比较三组结果

每组13个分片结束后可单独汇总；以下循环用于三组都结束后，各自严格验收（不加载模型、不需要GPU）：

```bash
N_ENVS=8
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  "$PY" scripts/evaluate_frozen.py --variant "$V" --home "$H" \
    --config "$REPO/resolved/${V}-70k.yaml" \
    --checkpoint "$H/runs/${V}-70k-s42/checkpoints/steps_70000_model.safetensors" \
    --sim-python "$SIM_PY" --output "$H/eval/${V}-70k-fixed-v2-n${N_ENVS}" \
    --plan-dir "$PLANS" --n-envs "$N_ENVS" --summarize
done
```

成功凭据是**三个独立的`FINAL_ACCEPTED.json`**：每组50份原始Boolean数组、2500条唯一trace，以及checkpoint/config/source/scene身份一致。平台SUCCEEDED、目录文件数齐全或smoke通过都不能替代该凭据。打印三组总成功率：

```bash
"$PY" - <<'PY'
import json, os
from pathlib import Path
home = Path(os.environ['H'])
scene_ids = set()
for group, variant in [('A', 'embedding3d_pos_a'), ('B', 'embedding3d_temporal_b'), ('C', 'query3d_temporal_c')]:
    path = home / 'eval' / f'{variant}-70k-fixed-v2-n8' / 'FINAL_ACCEPTED.json'
    d = json.loads(path.read_text())
    assert d['trials'] == 2500 and len(d['tasks']) == 50
    assert d['identity']['variant'] == variant
    scene_ids.add(d['episode_plan_set_sha256'])
    print(f"{group}\t{variant}\t{d['successes']}/{d['trials']}\t{100*d['successes']/d['trials']:.2f}%")
assert len(scene_ids) == 1, '三组不是同一套场景，不能作为同场景对照'
PY
```

若统一改过N_ENVS，上述目录后缀也需对应修改。保留每task的success数组、trace、失败回合和50任务分项，不只保存总分。三组此协议各2500回合，总计7500个计分回合；smoke不计入。与旧391/559等历史分数比较时另列场景协议，若需同场景基线/FIXED对照，应在同一PLANS重新评测对应权重。

## 8. 已验证与未验证

2026-10-24的C结构增补：7项定向CPU检查、2项A/B相邻检查及GitHub CPU CI通过；A/B父版本初始化state、RNG、输出一致。C尚无真实GPU训练/恢复/闭环成绩。本手册给出三组完整操作过程，不代表这些操作已在新集群执行。C证据见[定向检查记录](../provenance/QUERY50_TEMPORAL_C_VALIDATION.json)。

2026-10-23本次实测：

- Windows 68项检查中64通过、4项Linux专属跳过；Linux 68项全部通过。
- 单H100上A1500严格加载861keys、Bsmoke3严格加载885keys，分别完成n_envs=1/2各两个完整900步回合。合计8回合/7200模拟步/1440次完整50步动作对照；eager与graph逐元素完全一致，单/双环境的初始化与完整轨迹一致，CUDA与EGL UUID一致。
- 真实OpenDrawer演示204帧/1024点，在隔离目录重新生成8个数组，与原GT逐字节一致。复用历史环境时曾发现policy Python未装MuJoCo，因此该有界重建使用已安装MuJoCo3.3.1的sim Python；新安装脚本已包含该依赖，上方import检查不能省略。另一个缺少alignment凭据的旧样本输入被正确拒绝，随后使用真实既有凭据及实际源/缓存SHA核验后的输入，没有放宽校验。
- 最终训练启动代码核验961份源码；旧release配置会被拒绝；通过真实CUDA确定性矩阵前向/反向环境探针与正式配置dry-run。这不是新增完整模型训练结果。
- 补充的训练启动保护没有改变任何模型、GT或评测源码；最终发布与上述GPU运行版本的这些文件已逐项证明相同。

高效后端的正确性通过；本次smoke逐请求同时跑eager对照，**不是生产模式吞吐测试**。没有把baseline的12/24环境速度或利用率当作A/B性能，也没有据此宣称算法效果提升。

本次发布的精确源码和实际A/B加速验证见[发布验收记录](../provenance/POSITION_V3_HANDOFF_VALIDATION.json)。此前A1500、Bsmoke3的训练/完整恢复/动作推理、fslgpu eager闭环已经通过；这些历史证据不能自动替代本次新图后端的验收。

原SCO ACP曾发生NVIDIA EGL内部SIGABRT；容器capabilities=all也不能证明此问题已消失。fslgpu通过只对实测组合有效。新集群若出现原生崩溃、初始化SHA不一致或动作对照失败，保存任务身份/环境/栈并回交主对话，不自行修模型、改成功条件或删坏场景。

GitHub不含数据、GT大数组、权重和原训练完整状态。代码整理与有界GPU验收不等于在新集群从下载开始重跑全部数据和70k实验，也不等于方法性能提升。

`deployment/sco_position_v3_debug/`保留旧集群排障取证，不是新集群启动入口；新集群只使用本页的scripts/configs。
