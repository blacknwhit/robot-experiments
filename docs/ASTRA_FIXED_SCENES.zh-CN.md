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

**场景准备和正式评测使用同一套任务镜像、模拟器、驱动与渲染配置。** 在正式评测所用的作业环境中执行下面的准备命令，再让所有分片共享其`PLANS`。不要在另一种调试容器中准备后直接假定正式作业兼容；相同GPU型号、驱动版本和pip freeze也不足以证明初始图像完全相同。本次曾实际拦截过这种观测哈希不一致，原因未确定，没有放宽校验。遇到不一致应保留日志并停止，不删除或覆盖已封存清单，不把不同初始化结果混入同一集合。

## 2. 一次性准备固定场景，不跑模型

```bash
"$SIM_PY" scripts/scene_suite.py \
  --home "$HOME_DIR" --output "$PLANS" --egl-device-id 0 --workers 8
"$POLICY_PY" scripts/evaluate_frozen.py --plan-dir "$PLANS" --check-only
```

这一步需要一张可渲染的GPU，只创建环境并reset，不加载权重、不执行策略动作、不生成成功率；不需要先串行评测一遍基线。准备阶段独占其分配的资源，不与正式评测争用GPU。

完整50个task、2500个计分场景已在验收环境实际初始化并封存，[参考清单](../provenance/fixed_scene_reference_v2/PLAN_SET.json)及[验证记录](../provenance/FIXED_SCENE_VALIDATION.json)随仓库保存。正式评测使用目标集群按上述命令一次性生成的`PLANS`，其中绑定该环境的真实依赖与EGL观测哈希；以后所有模型及环境数复用同一目录。保持公开种子定义不变，也不要为每个checkpoint重新建集合。

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
| 同task环境数 | 默认`--n-envs 8`；允许1–50的整数 |
| 推理后端 | 自动使用`cuda-graph-pipeline`，原生batch=1 |
| 执行方式 | 8个独立模拟器异步供给；CPU预处理与GPU推理重叠 |
| OMP/MKL/OpenBLAS | 策略4/4/1，每个模拟器4/4/1，由入口设置 |
| H100资源起点 | 每卡约22 vCPU，预留足够内存，避免重CPU任务争用 |
| 数值/动作协议 | 原精度、TF32关闭、10次去噪、输出50步、每5步重规划 |
| 内容 | 每task固定episode0–49，原horizon与成功条件 |

这里改变的是场景初始化和执行调度。模型结构、权重、精度和成功条件不变。逻辑environment index固定为0，策略随机种子仍由task/episode/replan确定；物理worker编号不改变请求种子。

不要直接把旧`evaluate.py`的n_envs调大来代替本入口，不要叠加`--tasks-per-gpu`或`--rgb-last-frame`。此路径保留完整渲染，提速来源是并行供给和流水线。

## 4. 选择权重，直接开始正式评测

使用实际配置及权重路径。PI0.5选`baseline`，FIXED选`query_fixed`，配置中的`portable.variant`须对应；不同权重、配置或n_envs用不同`OUT`。

```bash
VARIANT=baseline
CONFIG=/absolute/path/to/baseline-resolved.yaml
CHECKPOINT=/absolute/path/to/trained-model.safetensors
OUT="$HOME_DIR/eval/baseline-70k-fixed2500-v2-n8"
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --plan-dir "$PLANS" --n-envs 8)

```

配置好上述变量后，直接执行下一节的正式分片命令。**不需要先跑`--smoke`，也不要求`SMOKE_ACCEPTED.json`。** 同一套固定场景可用于不同权重，准备好的`PLANS`保持不变。

每个策略进程仍自动严格加载完整模型，在首个真实请求比较原生与加速推理的完整动作；每次检查动作形状及有限值，每回合核对固定场景哈希。首请求对照使用正式请求的输入，不额外跑一批场景。任何检查失败都停止并保留证据。

输出目录通过`EVALUATION_IDENTITY.json`及锁绑定模型、权重、配置、源码、模拟器、场景集合和`n_envs`，并发分片必须一致。换权重、配置、源码、依赖或`n_envs`时使用新的`OUT`；场景环境未变时继续共用原`PLANS`。不同模型之间不要求动作相同。

`--smoke`保留为开发者显式选择的诊断工具，不属于普通正式评测的必经步骤。历史`legacy`入口的原有验收规则保持不变。

### 调整n_envs

可以设置3、6、12、16等1–50内的整数，不要求是2的幂，也不要求整除50。所有设置都恰好评测同一个task的episode0–49；worker编号不改变场景种子或策略请求种子，最终按episode编号汇总。修改`COMMON`中的`--n-envs`并使用新`OUT`即可，不重新准备场景、不新增完整smoke。

默认8是已有实测起点。更大并发会增加CPU、内存、显存及渲染竞争，未必更快；策略仍是一份模型、原生batch1，现有16个在途请求上限保持，更多环境会排队。1–50是接口允许范围，不表示每种硬件都能承载50个环境。

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

替换为其余49个task。此命令直接开始正式评测；物理worker按固定episode编号分配，完成顺序不影响内容及统计。

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

本次新协议已通过真实H100验收：PI0.5与Query-FIXED各自8个完整回合，单环境/八环境初始化、输入及执行动作轨迹、成功数组一致；完整50步动作共5760次零容差对照通过。以下速度使用baseline70k、CloseBlenderLid同8个场景，关闭逐请求eager重复计算；每行的轨迹仍与参考完全一致。

| 正常评测方式 | rollout秒数 | rollout平均GPU-util | 含环境初始化秒数 | 含初始化平均GPU-util |
|---|---:|---:|---:|---:|
| n_envs=4 | 154.11 | 58.34% | 190.34 | 47.57% |
| n_envs=8，第1次 | 102.58 | 89.36% | 136.01 | 68.12% |
| n_envs=8，第2次 | 104.95 | 84.25% | 139.11 | 63.37% |

八环境峰值显存22464MiB。上表来自同一已加载服务的有界闭环测试，模型加载另计；完整证据摘要见[验证记录](../provenance/FIXED_SCENE_VALIDATION.json)。**84–89%指这组场景的rollout均值，不是全50task、所有模型、含模型加载整项作业或连续3小时的保证。** 不用smoke时额外执行的eager计算冒充正式利用率。共享存储较慢时，可先将完全相同的权重和资产校验后缓存到节点本地SSD，减少加载期间空等。


CPU或显存不足可降到n_envs=4，保持`PLANS`并换新`OUT`直接运行。不要降低精度、减少去噪、缩短回合或重复无用推理来抬高利用率。
