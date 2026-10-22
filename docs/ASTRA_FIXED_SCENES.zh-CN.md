# 远端 Astra：固定新场景、单卡单task高吞吐评测

用户最新决定：不为复原历史场景重跑旧实验；以后所有新评测使用同一固定场景集合。旧391/2500、559/2500等数字单列为历史结果，不冒充新集合的逐场景配对结果。

远端只需更新仓库、运行现成入口，无需改代码。本文适用于已训练好的`baseline`（PI0.5）、`query_fixed`（最佳FIXED结构）及后续模型。示例不代表要求立即重测旧模型；只评测用户指定的权重。

## 1. 更新及路径

```bash
git pull --ff-only origin main
POLICY_PY=/absolute/path/to/policy/python
SIM_PY=/absolute/path/to/simulator/python
HOME_DIR=/absolute/path/to/robot-experiments-data
PLANS="$HOME_DIR/evaluation-scenes/fixed2500-v2"
```

沿用[主交接文档](../HANDOFF.zh-CN.md)准备的环境和资产。`HOME_DIR/receipts/sim.freeze.txt`必须是当前仿真环境的真实记录，不能复制旧集群的freeze冒充兼容。CUDA与EGL编号应映射到同一物理GPU；以下以两者均为0为例，容器/调度器重映射时使用已经核实的对应编号。

创建GPU容器/任务时设置 **`NVIDIA_DRIVER_CAPABILITIES=all`**，让计算与EGL渲染所需的驱动库一起暴露给容器；仅在已经启动的缺库容器内临时export不能补挂驱动库。`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`及逐卡`MUJOCO_EGL_DEVICE_ID`由以下入口设置。

## 2. 一次性准备固定场景，不跑模型

```bash
"$SIM_PY" scripts/scene_suite.py \
  --home "$HOME_DIR" --output "$PLANS" --egl-device-id 0 --workers 4
"$POLICY_PY" scripts/evaluate_frozen.py --plan-dir "$PLANS" --check-only
```

这一步需要一张可渲染的GPU，只创建环境并reset，不加载权重、不执行策略动作、不生成成功率；不需要先串行评测一遍基线。准备阶段独占其分配的资源，不与正式评测争用GPU。

- [固定种子清单](../assets/eval_scene_seeds_v1.json)规定50个task各50个计分场景；每task额外一个reset哨兵不计分。
- 每回合关闭旧环境并创建全新环境；构造前固定Python/NumPy种子，环境构造种子使用该回合的scene_seed。生成和评测共用同一初始化函数，避免只恢复随机数却残留前回合内部状态。新计划格式为`robocasa_seeded_episode_plan_v2`，不能复用早期未验收v1草案。
- 新协议同时固定RoboCasa台面候选区域的去重顺序：将按XML对象身份排列的`set`去重改为保留首次出现顺序，候选集合和均匀抽样权重不变。入口只对已核验源码的`Counter.get_reset_regions`做进程内修复；不改环境安装文件，不影响其它训练或旧评测进程。源码不匹配会明确拒绝，远端不要自行猜改。
- 保存初始观测（图像、语言、状态）及MuJoCo状态SHA，再封存`PLAN_SET.json`。完整检查必须显示`tasks:50`、`scored_episodes:2500`。
- 此后所有模型与n_envs共用同一个`PLANS`。不按checkpoint重新生成，不删除失败或慢场景。生成中断可重跑上述命令，已完整且身份一致的task会复用。
- 初始化内部采用短进程，每个进程最多创建8个独立场景，完整块写入`.chunks/`后才合并为task计划；种子和episode编号不变。保留块及退出码记录，中断后可复用完整块，不拼接不完整记录。
- 保存集合SHA。同一比较的`episode_plan_set_sha256`必须一致；相同种子表不代表已经证明任意集群/版本逐位相同。跨集群搬运仍须通过仿真环境及实际初始化SHA校验，不能绕过。

## 3. 正式推荐设置及实现变化

| 项目 | 设置 |
|---|---|
| 每张卡 | 一个task、一份模型 |
| 同task环境数 | `--n-envs 8` |
| 推理后端 | 自动使用`cuda-graph-pipeline`，原生batch=1 |
| 执行方式 | 8个独立模拟器异步供给；CPU预处理与GPU推理重叠 |
| OMP/MKL/OpenBLAS | 策略4/4/1，每个模拟器4/4/1，由入口设置 |
| H100资源起点 | 每卡约22 vCPU，预留足够内存，避免重CPU任务争用 |
| 数值/动作协议 | 原精度、TF32关闭、10次去噪、输出50步、每5步重规划 |
| 内容 | 每task固定episode0–49，原horizon与成功条件 |

这里改变的是场景初始化和执行调度。模型结构、权重、精度和成功条件不变。逻辑environment index固定为0，策略随机种子仍由task/episode/replan确定；物理worker编号不改变请求种子。

不要直接把旧`evaluate.py`的n_envs调大来代替本入口，不要叠加`--tasks-per-gpu`或`--rgb-last-frame`。此路径保留完整渲染，提速来源是并行供给和流水线。

## 4. 每套权重一次smoke

使用实际配置及权重路径。PI0.5选`baseline`，FIXED选`query_fixed`，配置中的`portable.variant`须对应；不同权重、配置或n_envs用不同`OUT`。

```bash
VARIANT=baseline
CONFIG=/absolute/path/to/baseline-resolved.yaml
CHECKPOINT=/absolute/path/to/trained-model.safetensors
OUT="$HOME_DIR/eval/baseline-70k-fixed2500-v2-n8"
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --plan-dir "$PLANS" --n-envs 8)

"$POLICY_PY" scripts/evaluate_frozen.py "${COMMON[@]}" \
  --smoke --gpu-ids 0 --egl-device-ids 0 --port-base 19090
```

入口先用当前模型单环境运行同8场景，再八环境运行，比较初始化、完整输入/执行动作轨迹和成功数组；每次请求还比较完整50步动作与eager输出，要求数值完全相等。不同模型之间不要求动作相同。

生成匹配身份的`SMOKE_ACCEPTED.json`后即可开始全量；不需要50个task分别smoke。更换权重、配置、n_envs或场景集合须新输出目录及新smoke。smoke不计入正式成绩。任何校验错误都保留日志并报告，不绕过门槛。

## 5. 全量：12个四卡分片＋1个两卡分片

分片使用`assets/eval_tasks.json`固定顺序。对PART=1…12各运行一个四卡作业，每张卡一个task：

```bash
PART=1
"$POLICY_PY" scripts/evaluate_frozen.py "${COMMON[@]}" \
  --part "$PART" --gpu-ids 0,1,2,3 --egl-device-ids 0,1,2,3 \
  --port-base "$((19090 + PART * 10))"

# 最后一个分片两张卡。
"$POLICY_PY" scripts/evaluate_frozen.py "${COMMON[@]}" \
  --part 13 --gpu-ids 0,1 --egl-device-ids 0,1 --port-base 19220
```

编号仅为作业内映射示例，按实际CUDA/EGL映射填写。各作业共享同一模型的`OUT`和`PLANS`；同宿主作业使用不同端口范围。不要把两个task分给同一物理卡。每task内部八环境，合计50回合，不取最先结束的50回合或改变场景分母。

若调度器使用单卡task数组，也可直接调用：

```bash
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --scene-protocol shared-scenes --task CloseBlenderLid \
  --gpu-id 0 --egl-device-id 0 --port 19090
```

替换为其余49个task。此命令使用同一`OUT`中已通过的smoke门槛；物理worker按固定episode编号分配，完成顺序不影响内容及统计。

## 6. 汇总及实际利用率

```bash
"$POLICY_PY" scripts/evaluate_frozen.py "${COMMON[@]}" --summarize
```

以`FINAL_ACCEPTED.json`为完整验收。保留50个raw JSON、2500条唯一轨迹、权重/配置/源码/场景身份和日志；报告成功数/2500、逐task结果、耗时及集合SHA。

作业内可在独立终端持续记录本作业分配GPU，结束后停止采样：

```bash
ALLOCATED_GPU_UUIDS=GPU-your-assigned-uuid  # 多卡填本作业已核实的UUID逗号列表
nvidia-smi --id "$ALLOCATED_GPU_UUIDS" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1 > "$OUT/gpu-util-part-${PART}.csv"
```

多个作业用不同采样文件，按真实GPU UUID统计，同时报告含模型加载/初始化的整体平均和rollout平均。smoke额外执行eager对照，不能用其利用率宣称正式效率。

此前H10080GB、baseline70k、CloseBlenderLid同8回合，八环境rollout约105秒、平均GPU-util84.01%/85.24%；含首次环境构建约137–139秒、65.77%/66.03%，模型加载另计。**这不是全50task、全部模型或连续3小时的保证，也不能代替本次新场景协议验收。**

CPU或显存不足可降到n_envs=4，保持`PLANS`、换新`OUT`并重新smoke。不要降低精度、减少去噪、缩短回合或重复无用推理来抬高利用率。
