# 远端 Astra：固定episode种子直接评测（2026-10-28）

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

新入口在每个仿真子进程内安装`mujoco331_native_rgb_owned_v1`：用 MuJoCo 3.3.1 官方 `Renderer` 管理离屏RGB绘制和读回，替换 robosuite 的旧离屏封装。物理 model/data、相机、场景选项、256×256图像、MSAA=4及Gym最终图像方向保持不变；输出缓冲区先清零，避免未写入时返回旧内存。回合切换时在资源所属EGL上下文内先释放 MuJoCo GPU资源，再关闭GL上下文，并恢复其它仍有效的上下文。自动循环GC延后到这些安全边界执行。

该适配器限定本入口的主线程RGB评测；不支持深度、分割或交互扰动。**每张卡一个策略服务、一个仿真进程，在该进程中同时保留n_envs个独立环境。** 仿真主线程负责物理step、渲染、重置和释放；线程池只负责各环境独立的策略网络请求，不能把env/GL交给网络线程。这样避开当前H100环境中多个仿真进程分别持有EGL时的异常路径，同时保留多环境请求流水线。执行版本为`single_process_envs_async_policy_v1`。

运行结果记录实际GLContext类型、MSAA和适配器版本。使用交接文档固定依赖和新Python进程，不叠加r12/r13的私有渲染补丁，不修改系统安装包。不需要远端Astra自行实现修复。正式入口调用`seeded_async_sim.py`；旧`seeded_episode_sim.py`命令也转到新入口，勿复制其中历史多进程辅助函数到远端启动脚本。

## 2. 固定的是每个episode的种子

```bash
"$POLICY_PY" scripts/evaluate_seeded.py --check-only
```

此命令仅检查仓库内的[种子定义](../assets/eval_scene_seeds_v1.json)和[任务清单](../assets/eval_tasks.json)，不创建模拟器、不生成计划、不加载模型。应显示50任务、2500个计分回合以及`reference_plan_required: false`。

- 每个task使用种子表前50个seed，对应固定episode编号0–49；旧第51个哨兵不参与本模式。
- 每回合在构造前重设相关RNG，使用该episode的seed新建环境，结束后关闭。保留稳定的场景候选顺序修复。
- 同进程的每个环境单独保存Python/NumPy RNG状态，构造/reset后捕获，每次完整step前恢复、step后保存；不是每步重新seed。环境自身的Generator仍由该环境独立拥有。策略网络线程不消费这些场景RNG。
- **不要把四个worker固定为seed7/8/9/10。** 场景seed由task＋全局episode编号决定，改变n_envs不会改变种子集合或计分回合。
- 策略随机数与场景seed分开：仍使用base7、task、逻辑环境0、全局episode编号、replan编号确定噪声，不使用物理worker编号。
- 不读取或比较预先保存的图像、state或轨迹哈希。本模式保证固定seed和统一生成流程，不宣称不同GPU/驱动逐像素一致。

## 3. 正式参数

| 参数 | 设置 |
|---|---|
| 同时任务 | 每GPU一个task、一份模型服务 |
| 每task环境数 | 默认4，可设1–50；n4已完成全50任务验收，n8仅做真实模型前缀短测；更大并发未在本入口完整验证 |
| 仿真进程 | 每task一个，主线程管理全部独立环境；异步网络线程连接同一策略服务 |
| 推理 | `cuda-graph-pipeline`，原生batch1 |
| 预测/执行 | 预测50步，每5个控制步重规划，10次去噪 |
| horizon | 直接使用任务清单已有值，**不再乘1.5** |
| task总墙钟超时 | 下方正式命令设置`--timeout 43200`（12小时），包括该task全部50回合；它不是单回合horizon |
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
  --n-envs "$N_ENVS" --timeout 43200)

"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --all-tasks --gpu-ids 0,1,2,3,4,5,6,7 --egl-device-ids 0,1,2,3,4,5,6,7 \
  --port-base 24090
```

每张卡顺序承担分配给它的task，同时只存在一个该入口的模型服务。n_envs=4时全机最多8个task、8个仿真进程、32个独立环境；所有50个task各跑固定50回合，共2500回合。不能取“最先完成的50个”，也不能遗漏或跳过报错回合。

`--timeout`只限制一个task的总墙钟时间，不改变任何回合控制步数。当前H100的长任务4回合测试约需20分钟，不能据此保证50回合能在脚本默认4小时内完成，因此正式命令显式给12小时预算；这不是已测得的50回合耗时。平台的作业时限还需覆盖一张卡顺序分配到的所有task。不要通过缩短horizon或跳过回合规避超时。

该参数不会改变集群自身的低利用率回收规则；本入口仍可能受CPU仿真限制，不能把较长超时或较大的n_envs当作高GPU利用率保证。

不同模型、权重、配置、代码、依赖记录或n_envs使用新的`OUT`。这个目录由`EVALUATION_IDENTITY.json`绑定运行身份；新协议结果不能混入旧r12/r13或PLANS模式输出。`query_fixed`等其它模型换对应VARIANT/config/checkpoint，保持同一任务和seed定义；本次不自动提交未被用户要求的模型评测。

单卡只测指定task：

```bash
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-ids 0 --egl-device-ids 0 --port-base 24090
```

如仍使用原来的4卡/2卡分片，`--part 1`至`--part 12`各配4卡，`--part 13`配2卡；每个分片共享同一模型的`OUT`。

```bash
PART=1
if [ "$PART" -eq 13 ]; then
  GPU_IDS=0,1
  EGL_IDS=0,1
else
  GPU_IDS=0,1,2,3
  EGL_IDS=0,1,2,3
fi
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" \
  --part "$PART" --gpu-ids "$GPU_IDS" --egl-device-ids "$EGL_IDS" \
  --port-base "$((24090 + PART * 10))"
```

## 5. 汇总、失败和恢复

`--all-tasks`成功结束会自动汇总，也可执行：

```bash
"$POLICY_PY" scripts/evaluate_seeded.py "${COMMON[@]}" --summarize
```

以`FINAL_ACCEPTED.json`为完整覆盖验收。它要求50份正式结果、2500个唯一task/episode组合、正确种子、原始Boolean成功数组及匹配运行身份；包含每份原始结果的SHA。它不宣称逐像素/轨迹相同。

完整且身份匹配的task可以复用。一个task中途失败时不会写成完整结果，也不会把异常计为任务失败分数或静默跳过；查清问题后重跑原命令，该未完整task按相同seed重跑。已有完整task不重复执行。

失败证据位于`OUT/runs/<task>/<attempt>/`：优先检查`ERROR.json`、`simulation.log`、`SIMULATOR_EXIT.json`，以及`simulation/SIMULATION_ERROR.json`、`simulation/CLEANUP_ERROR.json`。`simulation/SIMULATION_PROCESS.json`记录实际进程身份；`simulation/env*.initial.json`和`simulation/env*.progress.json`记录各环境的全局episode。观测异常会尽量保留`simulation/failed_observation.npz`。仿真进程启用faulthandler，原生崩溃可能只有栈和退出信号，没有Python异常JSON；SIGKILL通常没有可捕获的栈。模型日志在`server.log`。外层“task failed”不是根因。旧多进程失败归档的worker*.error.json/worker_exit.json属于历史入口，勿与新执行版本混淆。

## 6. n_envs与利用率参考

**当前入口先用`n_envs=4`。它已通过下方全50任务验证，但尚未达到70%–80%的GPU利用率目标。** 在本次H100的完整回合测试中，47/50个任务有完整rollout采样；这47项的任务平均GPU-util中位数为23.00%，范围14.54%–29.13%。另外3项存在采样缺口，未补算缺失时间。这里不是连续3小时或整项作业的平均值。

当前主线程承担CPU仿真与渲染，实际采样曾显示它在机器人控制器更新中运行而策略网络线程等待；这支持CPU侧限制，但不是完整的函数耗时剖析。`n_envs=8`通过了两次各128请求的真实模型短测，不能据此声称全任务更快或保证通过集群的低利用率回收规则。

旧固定场景入口的84%–85%、93%以及n12/n24/n37容量数字属于另一条执行路径，不能用于本入口承诺。历史原始记录保留在[旧H100容量证据](../provenance/H100_PI05_ENV_CAPACITY.json)。不要为了利用率重复无效计算、降低精度、缩短正式回合或改场景。

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

## 8. 本版已完成的验收与边界

2026-10-28在SCO H100 80GB、驱动580.95.05上，用PI0.5 baseline70k真实权重完成：

- **全部50个task × 每task固定episode 0–3，n_envs=4，共200个完整回合。** 全部跑满原horizon，共72,360次策略请求、361,800个控制步；没有使用预生成计划或参考图像/state/轨迹比较。
- 逐项复核种子、raw Boolean成功数组、结果与receipt SHA、813个严格加载权重键、代码/config/环境身份、退出状态及输入/动作有效性。最终写锁释放、评测进程退出、GPU显存归零。
- 两个先后使用的H100实例各通过98项Linux CPU测试和n8的128请求前缀检查。首次实例被外部暂停后，33份已完整且身份匹配的task结果经复核保留，其余17项在新实例完成；未复用半个回合，未改运行源码。

被测运行代码为`77836840709885d0ebde907d10ebd114b2b87a38`，source manifest为`ce30a84ebba6cee9e29d71906cb0e81d7d34ece136db4bd0d44670cd82e6d5c9`；之后只补交接文档与验收记录。逐任务证据和原始归档SHA见[本版验收记录](../provenance/SEED_ONLY_EVALUATION_VALIDATION.json)。

这是评测实现验收，**不是50×50正式成绩**。验收控制器为节省加载时间，在顺序task之间复用一份策略服务；仿真入口和模型服务使用原封存公开代码。上面的公开正式入口会按task启动服务。未实际跑过8张物理卡同时评测，也未直接测试远端Astra的535.129.03驱动；远端r13原生退出的精确原因仍未确定。私有RGB中间帧复用候选未启用。正常换模型无需重新做这套200回合验收，直接按正式命令运行并保留入口自带检查即可。
