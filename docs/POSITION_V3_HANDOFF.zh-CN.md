# A/B位置轨迹监督：新集群完整交接

本页是两版v3的执行入口，所有命令从仓库根目录运行。接手者只需配置路径、更新代码和执行，不需要重新实现GT、训练器或评测器。基线、Query-FIXED、v2仍保留，历史分数不作为本轮A/B结果。

## 1. 两版固定内容与脚本索引

| 项目 | A：`embedding3d_pos_a` | B：`embedding3d_temporal_b` |
|---|---|---|
| 母结构 | 原PI0.5动作prefix＋训练期Embedding | 同A |
| 点头 | PointMLP＋fusion MLP | 在fusion投影后增加2层时间attention |
| 时间交互 | 原MLP输出 | 同一点的后时刻读取自身及前时刻；不跨点；无RoPE |
| 目标 | 50步相对起点位置＋相邻差分 | 同A |
| 推理 | 不读3D输入，不执行点头 | 同A |

B的时间块为width256、8heads、FFN512、pre-LN残差、dropout0。完整结构和mask语义见[方法说明](POSITION_V3.zh-CN.md)。A/B的同形状权重与v2语义不同，不能互换恢复。

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
git clone https://github.com/blacknwhit/robot-experiments.git
cd robot-experiments
export ROBOT_EXPERIMENTS_HOME=/absolute/shared/robot-data
export H="$ROBOT_EXPERIMENTS_HOME"
mkdir -p "$H" resolved
python3 -m pip install uv
bash scripts/setup_policy.sh
bash scripts/setup_sim.sh
export PY="$H/envs/policy/bin/python"
export SIM_PY="$H/envs/sim/bin/python"
"$PY" -m unittest discover -s tests -v
sha256sum -c provenance/source_files.sha256
```

Linux、Python3.11、Torch2.7.1+cu128、Transformers4.53.2及对应独立overlay。安装器为四套runtime分别构建overlay，A/B共用trajectory_v3。策略和模拟器使用不同解释器，禁止把policy shim/preload混入模拟器。

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

## 3. GT只生成一套，A/B共用

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage prepare
# prepare只执行一次；接着提交8个CPU生成任务，各自I=0..7。
I=0
"$PY" scripts/prepare_gt.py --home "$H" --stage generate \
  --workers 6 --shard-index "$I" --num-shards 8
# 全部分片结束后再执行一次：
"$PY" scripts/prepare_gt.py --home "$H" --stage verify
```

生成器沿用已审计的`runtimes/trajectory_v2/starVLA/pri4r/`原始轨迹实现；A/B v3改变的是reader和监督，**不是重造另一套GT几何**。按NVIDIA固定revision下载states/XML/metadata，核对原始与训练数据的state/action/frame/timestamp，再重放记录状态，不积分预测动作。

1024个表面点：通常512机械臂＋512直接接触对象；无接触对象时1024机械臂；无背景。保留静止点，不用“运动大”筛选点。原始轨迹、刚体变换和有效性记录用于v3 reader：固定窗口首帧机器人base坐标，构造`p[t]-p[0]`和相邻差分；只监督连续有效前缀，缺帧/越界/尾部padding不当成零运动。GT未来点及mask只进loss。

资产路径为`$H/playground/Code/{robocasa365,robosuite}`，由setup_sim建立。`asset_root`属于配置SHA；新根重新生成时产生自己的SHA，configure会读取，不能硬填旧SHA。verify检查全部30,000个episode的来源、范围、数组哈希，写`data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json`；训练入口要求该凭据。

已有自有GT可整体搬运，须保留完整metadata/manifest/sampling_config并重新核验；不能只拷坐标数组或手写通过标记。旧完整训练的恢复还绑定旧代码与配置身份，本页新发布代码不支持假装与旧SCO运行逐位兼容；旧运行继续使用其封存release。

## 4. 在新集群验证训练、保存与恢复

先选择A，在一个实际双H100作业里顺序执行；B使用另一run-id重复。两版都是B32×2、acc1、global64，支持2/4/8卡同机配置，不能靠改变卡数直接恢复ZeRO状态。

```bash
V=embedding3d_pos_a  # B替换为embedding3d_temporal_b
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-smoke" \
  --gpus 2 --smoke --stop 2 --out "resolved/${V}-smoke2.yaml"
"$PY" scripts/import_preflight.py --config "resolved/${V}-smoke2.yaml"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/nccl_probe.py
"$PY" scripts/train.py --config "resolved/${V}-smoke2.yaml"
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-smoke" \
  --gpus 2 --smoke --stop 3 --out "resolved/${V}-smoke3.yaml"
"$PY" scripts/train.py --config "resolved/${V}-smoke3.yaml" --resume
```

检查实际GPU型号、两rank NCCL、steps1/2完整保存、新进程恢复后step3；每rank FP32主权重/Adam/BF16/RNG审计、有限loss/gradient、共享VLM/Embedding各层/点头（B还含时间块）更新。仅import通过不算GPU通过。完整checkpoint需要模型、两rank Adam/RNG、scheduler/data clocks、trajectory contract及`PORTABLE_COMPLETE.json`；残缺的新checkpoint跳过，没有完整状态则报错。

动作推理探针的参数见`"$PY" scripts/point_inference_probe.py --help`；它验证完整权重严格加载、无3D输入及joint/cache一致性。下面高效闭环入口另会严格加载所有模型key并禁止点头执行。

## 5. 短训与显式70k配置

短训使用全新run-id、原base/seed42，不能从smoke权重开始：

```bash
V=embedding3d_pos_a
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-short-s42" \
  --gpus 2 --engineering --stop 1500 --out "resolved/${V}-short.yaml"
"$PY" scripts/train_or_resume.py --config "resolved/${V}-short.yaml" --dry-run
"$PY" scripts/train_or_resume.py --config "resolved/${V}-short.yaml"
```

正式候选脚本已备好；下面命令仅在决定正式跑该候选时执行，本次发布没有自动提交70k任务：

```bash
V=embedding3d_pos_a  # B单独配置、独立run-id/作业
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-70k-s42" \
  --gpus 2 --formal --stop 70000 --out "resolved/${V}-70k.yaml"
"$PY" scripts/train_or_resume.py --config "resolved/${V}-70k.yaml" --dry-run
# 调度器的首次启动和每次重试都使用同一命令：
"$PY" scripts/train_or_resume.py --config "resolved/${V}-70k.yaml"
```

没有`--formal`不会把v3默认带入长训。`train_or_resume.py`首次新run启动base，run一旦存在就要求完整恢复；不创建重试循环、没有“恢复失败重新base”的兜底。恢复只能用相同world、相同代码/recipe，且持有唯一writer锁。SCO任务仍启用native autoRetrain=true、10分钟；其他调度器配置唯一重试owner。不要同时运行两个恢复管理器。

冻结候选：seed42、G8 ordered_random、global64/acc1、AdamW β0.9/0.95、eps1e-8、weight_decay1e-10、clip1、LR峰值2.5e-5/最低2.5e-6、warmup1000、scheduler时域100000，70k只是停止点。总loss为`action + ramp×(position + difference)`，辅助目标权重1.0、差分系数1.0、ramp前1000update线性升至1。这是当前工程候选，**没有证明该辅助量级最优**。首次update1额外完整保存，以后每1000和最终stop保存。

短训/正式都检查最初三条有限指标、预热40步后两个非重叠100步速度窗口和首个1k完整状态。loss下降不代表闭环效果提升。保留配置、freeze、日志、恢复审计和完整checkpoint。

## 6. 使用与参考聊天相同的高效测试

采用共享的`evaluate_frozen.py → evaluate_task.py → cuda-graph-pipeline`，每卡一个task、一份模型，多个独立模拟器异步供给；CPU准备与GPU推理重叠，模型仍batch1、原精度、10次去噪、输出50步、每5步重规划，完整渲染。**不是旧evaluate.py改大n_envs，也不减少去噪或渲染来提速。**

场景只准备一次，A/B/基线/FIXED共享同一套PLANS。必须在正式作业实际模拟器/镜像/渲染环境中准备；已存在一致PLANS直接复用，不为每套权重重建。

```bash
PLANS="$H/evaluation-scenes/fixed2500-v2"
"$SIM_PY" scripts/scene_suite.py --home "$H" --output "$PLANS" --egl-device-id 0 --workers 8
"$PY" scripts/evaluate_frozen.py --plan-dir "$PLANS" --check-only
```

必须显示50tasks/2500scored_episodes。固定seed7、任务清单、horizon、成功条件不改。新固定场景与旧391/559等历史成绩不是逐场景配对，不能直接冒称同一实验。

```bash
V=embedding3d_pos_a
N_ENVS=8  # A/B初始建议；H100 baseline测得的12/24容量不是A/B实测承诺
CONFIG="$PWD/resolved/${V}-70k.yaml"  # 短训则选择对应short配置
CHECKPOINT="$H/runs/${V}-70k-s42/checkpoints/steps_70000_model.safetensors"
OUT="$H/eval/${V}-70k-fixed-v2-n${N_ENVS}"
COMMON=(--variant "$V" --home "$H" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --sim-python "$SIM_PY" --output "$OUT" --plan-dir "$PLANS" --n-envs "$N_ENVS")
# 12个独立四卡作业，PART分别1..12；一个task一张卡：
PART=1
"$PY" scripts/evaluate_frozen.py "${COMMON[@]}" --part "$PART" \
  --gpu-ids 0,1,2,3 --egl-device-ids 0,1,2,3
# 最后一个双卡作业：
"$PY" scripts/evaluate_frozen.py "${COMMON[@]}" --part 13 --gpu-ids 0,1 --egl-device-ids 0,1
# 全部结束后：
"$PY" scripts/evaluate_frozen.py "${COMMON[@]}" --summarize
```

上面GPU编号只是默认枚举示例；任务内CUDA与EGL须核实物理UUID相同，重映射时传实际对应编号。分片在不同节点可共用端口基数；同机不同容器须按网络隔离情况设置独立`--port-base`。

普通同接口权重评测**不用每个task、每套权重重新跑完整smoke**。正式入口首个真实请求比较完整50步动作的eager与graph结果，要求逐元素完全一致；之后每请求检查形状/有限值，每回合核对初始化观测/物理状态哈希。任何失败停止并保留日志，不降级或放宽校验。开发者`--smoke`仍可显式选用以比较n1/nN并逐请求对照。

`n_envs`支持1–50，A/B先从8起，根据显存、CPU、内存选择；接口范围不是硬件容量保证。换权重、配置、代码、依赖或n_envs必须新OUT，场景环境不变则继续复用PLANS。不要叠加tasks-per-gpu/rgb-last-frame。推理不执行3D点头，但不能用baseline的显存/速度数字冒充A/B实测。

成功条件为`FINAL_ACCEPTED.json`：50份原始Boolean数组、2500条唯一trace、完整checkpoint/config/source/scene身份一致；平台SUCCEEDED或文件数量齐全不够。A/B当前不存在正式70k成绩。

## 7. 已验证与未验证

本次发布的精确源码和实际A/B加速验证见[发布验收记录](../provenance/POSITION_V3_HANDOFF_VALIDATION.json)。此前A1500、Bsmoke3的训练/完整恢复/动作推理、fslgpu eager闭环已经通过；这些历史证据不能自动替代本次新图后端的验收。

原SCO ACP曾发生NVIDIA EGL内部SIGABRT；容器capabilities=all也不能证明此问题已消失。fslgpu通过只对实测组合有效。新集群若出现原生崩溃、初始化SHA不一致或动作对照失败，保存任务身份/环境/栈并回交主对话，不自行修模型、改成功条件或删坏场景。

GitHub不含数据、GT大数组、权重和原训练完整状态。代码整理与有界GPU验收不等于在新集群从下载开始重跑全部数据和70k实验，也不等于方法性能提升。
