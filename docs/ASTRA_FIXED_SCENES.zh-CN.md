# 远端 Astra：固定2500回合的高吞吐评测

> 本页对应已完成H100全50任务实现验收的并行入口。正式评测直接更新main并执行下方命令；固定50×50种子，无需预生成计划或另跑完整smoke。

## 只需更新并运行

远端Astra负责更新仓库、填写本机路径并运行现成入口，无需自行修改代码、生成场景计划或再次完成一套smoke。当前入口是 **`scripts/evaluate_parallel.py`**：每卡一份常驻模型，多task共用一组CPU仿真worker，每卡一个EGL渲染进程。不要混入旧r12/r13私有补丁，也不要继续使用旧单进程入口来期待同样的吞吐。

不同模型的variant、源码/配置映射和GPU验证边界见[多版本模型说明](EVALUATION_MODELS.zh-CN.md)。已有7种`--variant`继续使用下方命令；新增结构可用`--model-spec /path/model.json`替代`--variant/--config/--checkpoint`，完整命令和模型适配接口见该文档。以后模型迭代修改模型自己的配置/adapter，无需修改本评测调度、仿真和计分代码。登记或接口通过不代表所有结构都已完成同等GPU验证。

**本页正式配置按单机8×H100 SXM 80GB编写。** 沿用此前的188vCPU、约1600GiB内存申请即可作为起点；这是用户现有资源配置，不是实测的最低资源要求。

| 8卡任务的实际配置 | 设置 |
|---|---|
| 启动方式 | 整机只启动下方一个调度入口；它自动创建8路GPU服务 |
| 模型副本 | 每卡1份，共8份；同一卡多个task共享这一份 |
| 物理仿真 | 每卡16个CPU worker，共最多128个 |
| 渲染进程 | 每卡1个EGL拥有者，共8个；每个活跃环境仍有自己的原生渲染资源 |
| 初始任务分配 | 每卡6个task，按horizon估计均衡；剩余2个由先完成者领取 |
| 模型与仿真 | 各卡独立运行，不做8卡DDP/NCCL同步推理 |
| 容器创建环境变量 | `NVIDIA_DRIVER_CAPABILITIES=all` |
| 渲染环境变量 | 入口自动设置`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`和各卡的`MUJOCO_EGL_DEVICE_ID` |
| 线程 | 策略每进程OMP/MKL=4、OpenBLAS=1；CPU仿真每进程OMP/MKL/OpenBLAS=1，由入口设置 |
| 同时评测的模型 | 先完成一版，再换VARIANT/config/checkpoint及新OUT测下一版，复用同一2500种子表 |

不要再套8份外层runner或与旧r13进程同时占用这8张卡；一个入口已经会使用全部8卡。GPU/EGL使用平台已核对的物理映射，不能把8个渲染进程都指定为EGL0。

```bash
git pull --ff-only origin main

POLICY_PY=/absolute/path/to/policy/python
SIM_PY=/absolute/path/to/simulator/python
HOME_DIR=/absolute/path/to/robot-experiments-data
VARIANT=baseline
CONFIG=/absolute/path/to/baseline-resolved.yaml
CHECKPOINT=/absolute/path/to/trained-model.safetensors
OUT="$HOME_DIR/eval/${VARIANT}-70k-parallel-seeds-v1"

"$POLICY_PY" scripts/evaluate_parallel.py \
  --variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --all-tasks --gpu-ids 0,1,2,3,4,5,6,7 \
  --egl-device-ids 0,1,2,3,4,5,6,7 --port-base 24090 \
  --n-envs 16 --tasks-per-gpu 50 --simulator-cpu-threads 1 \
  --rgb-schedule final --initial-rgb deferred --timeout 172800
```

先按[主交接文档](../HANDOFF.zh-CN.md)安装环境、获取资源并准备模型config。保留真实的 `$HOME_DIR/receipts/sim.freeze.txt`，模拟器必须是固定依赖的MuJoCo3.3.1/RoboCasa环境。`SIM_PY`可以是已验证的解释器包装脚本。容器**创建时**设置 `NVIDIA_DRIVER_CAPABILITIES=all`。

GPU和EGL编号必须实际指向同一物理GPU；示例0–7仅适用于两者顺序一致的机器。不要把策略内部的`cuda:0`当宿主EGL0。单卡、2卡、4卡只需相应缩短两组编号，例如2卡都填`0,1`；不用手工分50个task，入口自动分配。

正式运行不要传任何 `--diagnostic-*` 参数，不传 `--plan-dir`，不运行 `scene_suite.py`。不要截短horizon或按最先完成的回合计分。**保留示例中的n16/tasks50/final/deferred显式参数**，它们不是CLI默认值；省略后不会自动采用已测的高吞吐配置。可选 `--dry-run` 只打印种子覆盖和调度参数，不加载模型或运行场景。

## 单卡参数与多模型初始场景复核（2026-10-29）

最新model-spec接口已用PI0.5与Query-FIXED真实权重检查。4task×4seed共16对初始MuJoCo/RNG状态、RGB和机器人状态全部一致；不把该抽样说成2500个场景全部逐像素比较。正式模式仍无参考图片hash门槛。

单H10080GB/22vCPU、**PI0.5 baseline70k**的同工作量短测，n_envs4/8/16分别用时171.25/126.59/115.53秒；**建议从16开始**，每卡仍只加载一份模型，多个task共享环境池。n16首请求→末步GPU-util89.75%，含仿真启动/场景初始化的整段30.49%；两者都不包含共享模型加载/公共预热。短测只有每episode40次请求，不能拿89.75%当完整评测或3小时回收窗口保证，也不能直接套用到FIXED或其他结构。[完整表格、模型接入和证据边界](EVALUATION_MODELS.zh-CN.md)。

## 并发设置含义

| 参数 | 建议及含义 |
|---|---|
| n_envs | **每GPU总计16个**独立CPU仿真worker，不是每task16个 |
| tasks-per-gpu | 每批上限50个task；单卡让全50项共用worker池，2/4/8卡初批各25/12/6项，按horizon均衡 |
| 调度 | 按horizon估计首批工作量并均衡分配；一批完成后领取下一批，模型不重新加载 |
| 推理 | `cuda-graph-pipeline`，batch1；请求流水线，无需增加模型副本 |
| CPU线程 | 策略OMP/MKL4、OpenBLAS1；各仿真worker OMP/MKL1、OpenBLAS1 |
| RGB | 保留策略实际读取的三路RGB；省略未被读取的中间绘制及构造期丢弃绘制 |
| 预测/执行 | 预测50步、每5个控制步重规划、10次去噪 |
| horizon | 直接使用任务清单已有值，不乘1.5、不缩短 |
| 精度 | 原混合精度，TF32关闭 |
| timeout | 每批task的总墙钟上限172800秒（48小时），不是单回合长度 |

8卡时最多128个CPU物理worker，另有8份模型和8个渲染进程。此前的188vCPU配置可以从每卡16起步；不同CPU/内存配额需调整。实际验证的平台为单H10080GB、22vCPU/240GiB，不能把8卡吞吐直接乘8当成实测。

允许把n_envs改为8、12等值，**种子和2500回合集合不变**。较大值不必然更快；本次短测最终配置n16快于n8，但没有证明16是所有机器的全局最优。每次改变并发、代码、权重或配置使用新OUT。8卡50任务会先各领6个，剩余2个由先完成者继续领取；尾部任务变少时部分卡可能空闲，这是有限工作量的尾部，不应额外做无效计算抬高利用率。

`--timeout`只限制批次墙钟。单卡整批2500回合可能很长，因此示例给48小时上限；这不是实测耗时承诺。平台总时限仍须覆盖实际评测。不要通过减少回合/物理步绕过作业时限。

## 永久固定的2500个seed

协议为 `fixed_episode_seeds_v1`，执行方式为 `parallel_physics_single_render_v1`。

- 50个task，每个固定episode编号0–49；seed来自仓库 `assets/eval_scene_seeds_v1.json` 的前50项。旧第51个哨兵不计分。
- seed文件SHA256：`b1c5243b89f4d6283f01cb950bb93911269e8b9068f334062a869d604dca1c9b`。
- scene seed由task和全局episode编号确定，不受GPU、worker、完成顺序或n_envs影响。不要把worker直接固定为7/8/9/10。
- 策略噪声仍由base7、task、逻辑环境0、全局episode和replan编号确定；物理worker编号不参与。
- 不预生成2500场景计划，不与封存初始图片/state/轨迹哈希比较。保留相同依赖和生成逻辑；固定seed不等于跨驱动逐像素保证。
- baseline、query_fixed及后续兼容模型使用同一份任务/seed定义。更换模型只换VARIANT/config/checkpoint/OUT，无需重新生成seed或重新做完整验收。

## 正确完成、汇总及恢复

`--all-tasks`结束会自动汇总。正式成功标记为 `OUT/FINAL_ACCEPTED.json`，必须覆盖 **50个task、2500个唯一task/episode组合**，包含匹配身份、seed、原始Boolean成功数组和结果SHA。只有100回合的开发验收不等于正式成绩。

每个task结果在`OUT/formal/`，模型日志在`OUT/services/<id>/`，批次及worker证据在`OUT/parallel-runs/<id>/`。保留模型严格加载、首个真实请求eager/Graph对照、RGB/状态/动作有效性及覆盖检查；它们不是已经取消的封存场景哈希门槛。

**已取消的严格比较**：预生成2500场景计划，以及初始图片、MuJoCo状态、完整轨迹与封存参考的逐位/哈希一致性检查。正式运行不会因为与旧参考相差几个像素而退出。

**仍保留的必要检查**：模型权重完整加载；每个策略服务的首个真实请求核对eager/Graph动作，防止加速实现改变模型计算；非黑且有效的RGB、有限状态/动作；固定种子及2500回合覆盖。首请求检查不是额外smoke，也不重跑一套场景。硬件/驱动、依赖、资源或真实程序错误仍可能报错；不能承诺任何远端机器零错误，更不能把错误回合当作有效成绩。当前已验证的是本页记录的H100环境和全部50种任务流程。

发生错误时入口退出并清理自己创建的进程。优先查看批次`simulation.log`、`EXIT.json`以及`simulation/ERROR.json`、`CLEANUP_ERROR.json`、`worker_exit.json`和worker日志。原生崩溃可能只有信号/faulthandler栈，SIGKILL可能没有Python错误JSON；外层“task failed”不足以判断原因。不要把异常当成失败分数或跳过该seed。若出现全黑观测，或单卡正常但多卡失败，先按[全黑观测取证说明](BLACK_FRAME_DIAGNOSIS.zh-CN.md)读取已有产物；不需要重跑全量或自行修改评测代码。

修复外部问题后可重跑**完全相同命令和OUT**。入口只复用有完整提交标记、raw SHA、正常退出且身份匹配的task；未提交的一批会保留证据后按原seed重跑。大任务池的代价是提交粒度更粗：只有整批核验通过才提交，批内晚期错误可能需要重跑该批全部task。若集群频繁抢占，可在新OUT中将tasks-per-gpu降到8换取更细恢复粒度，但不保证同样的速度。一个输出目录只允许运行一个调度入口，不能拆成多个并发命令共同写入。若要单独测任务，用新OUT及 `--tasks WeighIngredients,PackIdenticalLunches` 替换`--all-tasks`，每项仍跑固定50回合。

已有全套结果需要重新汇总时，把同一命令中的`--all-tasks`替换为`--summarize`，其余身份参数保留。

## 实际性能及验收范围

同一张H10080GB/580.95.05、baseline70k，相同1280请求：旧单进程n4为936.27秒；新并行n16＋final/deferred为158.52秒，约 **5.91倍吞吐**。两组均包含仿真启动、reset及清理，排除共用模型加载；单次短测不能外推所有task。

新配置4800请求长测为367.91秒：**全仿真窗口平均71.07%，首个真实请求至最后回合完成平均89.37%**。这是已测窗口，不是连续3小时、8卡整作业或任意新机器保证。详细对照、逐轨迹/RNG验证及实现边界见 [技术与性能说明](PARALLEL_SEEDED_EVALUATION.zh-CN.md)。

最终公开入口已在一张H10080GB、22vCPU/240GiB、驱动580.95.05上完成 **全50个task × 每项episode0/1，共100个完整回合**。使用本页n16/tasks50/final/deferred参数，未截短任何horizon，共36,180次策略请求、180,900个控制步。该卡与前期两task调优使用的物理H100不同。

| 实际窗口 | 耗时 | GPU-util采样均值 |
|---|---:|---:|
| 公开命令启动至退出，含模型加载、初始化、重置及收尾 | 44.67分钟 | **72.04%** |
| 首次真实策略请求至最后回合完成，包含中途重置 | 39.10分钟 | **82.16%** |

同卡相同100回合，先前每批8task为49.38分钟、整程64.70%；扩大任务池后为44.67分钟、72.04%，本次耗时减少约9.53%。不是8物理卡实测，也不是连续3小时保证。1756条整程样本全部保留；其中有一个5.19秒采样间隔（其余不超过2.09秒），此间隔约占整程0.194%，原始质量标记已公开，没有裁掉低利用率阶段。

逐原始episode/合并结果、固定seed、horizon、813个权重键、源码/依赖身份、正常worker退出及最终写锁释放均已审核。两轮不同任务池下的100个初始观测数组完全一致，100个最终成功Boolean一致；这是开发对照证据，不给后续模型增加参考图片校验要求，也不是完整轨迹逐帧证明。最终109项Linux CPU检查（1跳过）及10项原生模拟器检查通过，所有评测进程退出、显存归零。

运行源码为`80efbe4c1487ab32107c45876470ef1fc34ac63e`，980项runtime manifest为`77b9bf5138b6a105d14d03f15731f3602f4397332351727434cef0530d3926a0`；之后只补文档与证据。逐任务原始结果SHA、采样窗口和归档信息见[最终验收记录](../provenance/PARALLEL_EVALUATION_VALIDATION.json)，短测记录见[调优证据](../provenance/PARALLEL_EVALUATION_TUNING.json)。

**100回合是覆盖全部任务的实现验收，不是完整2500回合正式成绩。** 上面的正式命令会跑固定2500回合。远端535.129.03驱动尚未直接验证；这里不宣称已经定位旧r13的精确原生退出根因。

历史单进程结果见 `provenance/SEED_ONLY_EVALUATION_VALIDATION.json`；旧PLANS说明见 [历史协议](ASTRA_RECORDED_PLANS_LEGACY.zh-CN.md)。这些记录保留，不能用历史84%–85%等数字替代本入口的实际采样。
