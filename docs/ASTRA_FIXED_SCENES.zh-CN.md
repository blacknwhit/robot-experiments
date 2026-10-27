# 远端 Astra：固定episode种子直接评测（2026-10-27）

当前使用 **`fixed_episode_seeds_v1`** 协议。用户已决定取消预生成2500场景计划及初始图像/状态参考哈希对照。远端只需更新仓库并运行下面的现成入口，不再运行`scene_suite.py`，不传`--plan-dir`，不需要完整smoke或`SMOKE_ACCEPTED.json`。

这不是关闭所有错误检查：仍严格加载模型、检查观测和动作有效性、记录失败并核验50×50覆盖。不会根据“分数接近预期”自动认定实现正确，也不比较不同模型的轨迹。

## 1. 更新与环境

```bash
git pull --ff-only origin main
POLICY_PY=/absolute/path/to/policy/python
SIM_PY=/absolute/path/to/simulator/python
HOME_DIR=/absolute/path/to/robot-experiments-data
```

沿用[主交接文档](../HANDOFF.zh-CN.md)配置环境、模型和资源；保留真实的`$HOME_DIR/receipts/sim.freeze.txt`。不需要从旧r12/r13复制评测脚本，也不需要删除或覆盖旧的封存计划。

容器/任务创建时设置`NVIDIA_DRIVER_CAPABILITIES=all`。入口使用EGL并按传入的物理设备映射绑定模拟器；CUDA与EGL编号必须对应同一张物理卡。不能默认CUDA内部的`cuda:0`等于宿主EGL设备0。策略与模拟器使用隔离的依赖路径；模拟器不会继承策略的NCCL预加载。

新入口自动安装进程内的`owned_egl_context_episode_gc_v1`修复：释放旧渲染器时绑定其自己的EGL上下文，随后恢复仍在使用的上下文；不再因释放旧上下文而解绑另一个渲染器，并在原生绘制/读回前重新绑定。循环垃圾回收延后到回合关闭后执行，普通引用计数释放不变，避免旧原生渲染资源在新上下文构造/绘制中被回收。相机、MSAA、图像尺寸、渲染格式不变。它针对交接环境固定的robosuite版本校验原函数SHA，不改全局安装文件。不要叠加r12/r13的本地EGL补丁；若提示`Unsupported robosuite EGL method`，使用交接文档固定依赖和本仓库入口，不能跳过校验套用未知补丁。

## 2. 固定的是每个episode的种子

```bash
"$POLICY_PY" scripts/evaluate_seeded.py --check-only
```

此命令仅检查仓库内的[种子定义](../assets/eval_scene_seeds_v1.json)和[任务清单](../assets/eval_tasks.json)，不创建模拟器、不生成计划、不加载模型。应显示50任务、2500个计分回合以及`reference_plan_required: false`。

- 每个task使用种子表前50个seed，对应固定episode编号0–49；旧第51个哨兵不参与本模式。
- 每回合在构造前重设相关RNG，使用该episode的seed新建环境，结束后关闭。保留稳定的场景候选顺序修复。
- **不要把四个worker固定为seed7/8/9/10。** 场景seed由task＋全局episode编号决定，改变n_envs不会改变种子集合或计分回合。
- 策略随机数与场景seed分开：仍使用base7、task、逻辑环境0、全局episode编号、replan编号确定噪声，不使用物理worker编号。
- 不读取或比较预先保存的图像、state或轨迹哈希。本模式保证固定seed和统一生成流程，不宣称不同GPU/驱动逐像素一致。

## 3. 正式参数

| 参数 | 设置 |
|---|---|
| 同时任务 | 每GPU一个task、一份模型服务 |
| 每task环境数 | 默认4，可设1–50；H100 PI0.5资源充足时可参考12或24 |
| 推理 | `cuda-graph-pipeline`，原生batch1 |
| 预测/执行 | 预测50步，每5个控制步重规划，10次去噪 |
| horizon | 直接使用任务清单已有值，**不再乘1.5** |
| 精度 | 原模型混合精度，TF32关闭 |
| CPU线程 | 策略/仿真每进程OMP4、MKL4、OpenBLAS1 |
| 校验 | 严格权重加载、首个真实请求eager/Graph对照、RGB形状/范围/全黑检查、有限状态/动作、回合覆盖 |
| 不再要求 | 预生成PLANS、参考像素/state/轨迹完全一致、每权重完整smoke |

首请求eager/Graph对照只核对加速推理是否与原生推理相符，使用真实请求，不额外跑场景；后续请求不重复eager计算。无需关闭它来取消场景哈希比较。

## 4. 8卡正式评测：每卡一个task

以下参数与用户当前8×H100、每卡4环境的设置对应。将权重/config等替换为真实路径。

```bash
VARIANT=baseline
N_ENVS=4
CONFIG=/absolute/path/to/baseline-resolved.yaml
CHECKPOINT=/absolute/path/to/trained-model.safetensors
OUT="$HOME_DIR/eval/${VARIANT}-70k-seed-only-v1-n${N_ENVS}"
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --n-envs "$N_ENVS")

"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --all-tasks --gpu-ids 0,1,2,3,4,5,6,7 --egl-device-ids 0,1,2,3,4,5,6,7 \
  --port-base 24090
```

每张卡顺序承担分配给它的task，同时只存在一个该入口的模型服务。全机最多8个task和32个仿真环境；所有50个task各跑固定50回合，共2500回合。不能取“最先完成的50个”，也不能遗漏或跳过报错回合。

不同模型、权重、配置、代码、依赖记录或n_envs使用新的`OUT`。这个目录由`EVALUATION_IDENTITY.json`绑定运行身份；新协议结果不能混入旧r12/r13或PLANS模式输出。`query_fixed`等其它模型换对应VARIANT/config/checkpoint，保持同一任务和seed定义；本次不自动提交未被用户要求的模型评测。

单卡只测指定task：

```bash
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-ids 0 --egl-device-ids 0 --port-base 24090
```

如仍使用原来的4卡/2卡分片，`--part 1`至`--part 12`各配4卡，`--part 13`配2卡；每个分片共享同一模型的`OUT`。

```bash
PART=1
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --part "$PART" --gpu-ids 0,1,2,3 --egl-device-ids 0,1,2,3 \
  --port-base "$((24090 + PART * 10))"
```

## 5. 汇总、失败和恢复

`--all-tasks`成功结束会自动汇总，也可执行：

```bash
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" --summarize
```

以`FINAL_ACCEPTED.json`为完整覆盖验收。它要求50份正式结果、2500个唯一task/episode组合、正确种子、原始Boolean成功数组及匹配运行身份；包含每份原始结果的SHA。它不宣称逐像素/轨迹相同。

完整且身份匹配的task可以复用。一个task中途失败时不会写成完整结果，也不会把异常计为任务失败分数或静默跳过；查清问题后重跑原命令，该未完整task按相同seed重跑。已有完整task不重复执行。

失败证据位于`OUT/runs/<task>/<attempt>/`：优先检查`ERROR.json`、`simulation.log`、`simulation/worker*.log`、`worker*.error.json`和`worker_exit.json`。每个worker独立保存stdout/stderr并启用原生信号栈；`WORKER_FAILURE.json`保存发现异常时、清理其它worker之前的退出码，避免把后续SIGTERM误当根因。原生崩溃可能没有Python异常JSON；SIGKILL通常也没有可捕获的栈。观测异常会尽量保留`worker*.failed_observation.npz`。模型日志在`server.log`。外层“task failed”不是根因。

## 6. n_envs与利用率参考

之前SCO H10080GB/22vCPU/240GiB、PI0.5 baseline70k的40-replan容量短测中，n12/n24/n32均约93.4%–93.6% rollout GPU-util、15请求/秒；n24单次均值最高93.64%，n12为93.52%且显存更低（28.23对50.43GiB）。这是旧固定场景入口的短测数据，不是本次新入口的完整benchmark测量，也不保证不同task/模型或连续3小时平均值。

n37短测曾通过，n38触发4GiB显存余量保护，不能当成其它task的默认并发。当前先用4；资源和稳定性允许时，可按需改12或24，并换新OUT。不要为了利用率重复无效计算、降低精度、缩短正式回合或改场景。

## 7. 开发者诊断模式（普通正式评测无需运行）

```bash
"$POLICY_PY" scripts/evaluate_seeded.py --variant baseline --home "$HOME_DIR" \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" \
  --output "$HOME_DIR/eval/seed-only-diagnostic" --task PackIdenticalLunches \
  --gpu-ids 0 --egl-device-ids 0 --n-envs 2 --diagnostic-episodes 0,3 \
  --diagnostic-max-replans 8 --port-base 24590
```

诊断使用原horizon，仅在外层停止有限请求，未完成回合不产生成功率汇总。诊断结果与正式结果目录、身份分开，不能生成2500回合验收。普通换checkpoint无需先运行此模式。

旧的像素哈希对照协议保留在[历史PLANS执行说明](ASTRA_RECORDED_PLANS_LEGACY.zh-CN.md)；既有封存文件和历史验收记录未删除或改写。
