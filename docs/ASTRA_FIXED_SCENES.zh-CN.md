# 远端 Astra：固定2500回合的高吞吐评测

> 发布验收进行中：该分支的最终全50任务门槛尚未结束。合入main后按本页直接运行；不要从开发分支启动正式评测。

## 只需更新并运行

远端Astra负责更新仓库、填写本机路径并运行现成入口，无需自行修改代码、生成场景计划或再次完成一套smoke。当前入口是 **`scripts/evaluate_parallel.py`**：每卡一份常驻模型，多task共用一组CPU仿真worker，每卡一个EGL渲染进程。不要混入旧r12/r13私有补丁，也不要继续使用旧单进程入口来期待同样的吞吐。

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
  --n-envs 16 --tasks-per-gpu 8 --simulator-cpu-threads 1 \
  --rgb-schedule final --initial-rgb deferred --timeout 43200
```

先按[主交接文档](../HANDOFF.zh-CN.md)安装环境、获取资源并准备模型config。保留真实的 `$HOME_DIR/receipts/sim.freeze.txt`，模拟器必须是固定依赖的MuJoCo3.3.1/RoboCasa环境。`SIM_PY`可以是已验证的解释器包装脚本。容器**创建时**设置 `NVIDIA_DRIVER_CAPABILITIES=all`。

GPU和EGL编号必须实际指向同一物理GPU；示例0–7仅适用于两者顺序一致的机器。不要把策略内部的`cuda:0`当宿主EGL0。单卡、2卡、4卡只需相应缩短两组编号，例如2卡都填`0,1`；不用手工分50个task，入口自动分配。

正式运行不要传任何 `--diagnostic-*` 参数，不传 `--plan-dir`，不运行 `scene_suite.py`。不要截短horizon或按最先完成的回合计分。**保留示例中的n16/tasks8/final/deferred显式参数**，它们不是CLI默认值；省略后不会自动采用已测的高吞吐配置。可选 `--dry-run` 只打印种子覆盖和调度参数，不加载模型或运行场景。

## 并发设置含义

| 参数 | 建议及含义 |
|---|---|
| n_envs | **每GPU总计16个**独立CPU仿真worker，不是每task16个 |
| tasks-per-gpu | 每批最多8个task共享上述worker及一份模型；按GPU数和剩余任务自动缩小 |
| 调度 | 按horizon估计首批工作量并均衡分配；一批完成后领取下一批，模型不重新加载 |
| 推理 | `cuda-graph-pipeline`，batch1；请求流水线，无需增加模型副本 |
| CPU线程 | 策略OMP/MKL4、OpenBLAS1；各仿真worker OMP/MKL1、OpenBLAS1 |
| RGB | 保留策略实际读取的三路RGB；省略未被读取的中间绘制及构造期丢弃绘制 |
| 预测/执行 | 预测50步、每5个控制步重规划、10次去噪 |
| horizon | 直接使用任务清单已有值，不乘1.5、不缩短 |
| 精度 | 原混合精度，TF32关闭 |
| timeout | 每批task的总墙钟上限43200秒，不是单回合长度 |

8卡时最多128个CPU物理worker，另有8份模型和8个渲染进程。此前的188vCPU配置可以从每卡16起步；不同CPU/内存配额需调整。实际验证的平台为单H10080GB、22vCPU/240GiB，不能把8卡吞吐直接乘8当成实测。

允许把n_envs改为8、12等值，**种子和2500回合集合不变**。较大值不必然更快；本次短测最终配置n16快于n8，但没有证明16是所有机器的全局最优。每次改变并发、代码、权重或配置使用新OUT。8卡50任务会先各领6个，剩余2个由先完成者继续领取；尾部任务变少时部分卡可能空闲，这是有限工作量的尾部，不应额外做无效计算抬高利用率。

`--timeout`只限制批次墙钟，平台总时限还需覆盖该卡可能领取的全部批次。不要通过减少回合/物理步来绕过作业时限。

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

发生错误时入口退出并清理自己创建的进程。优先查看批次`simulation.log`、`EXIT.json`以及`simulation/ERROR.json`、`CLEANUP_ERROR.json`、`worker_exit.json`和worker日志。原生崩溃可能只有信号/faulthandler栈，SIGKILL可能没有Python错误JSON；外层“task failed”不足以判断原因。不要把异常当成失败分数或跳过该seed。

修复外部问题后可重跑**完全相同命令和OUT**。入口只复用有完整提交标记、raw SHA、正常退出且身份匹配的task；未提交的一批会保留证据后按原seed重跑。一个输出目录只允许运行一个调度入口，不能拆成多个并发命令共同写入。若要单独测任务，用新OUT及 `--tasks WeighIngredients,PackIdenticalLunches` 替换`--all-tasks`，每项仍跑固定50回合。

已有全套结果需要重新汇总时，把同一命令中的`--all-tasks`替换为`--summarize`，其余身份参数保留。

## 实际性能及验收范围

同一张H10080GB/580.95.05、baseline70k，相同1280请求：旧单进程n4为936.27秒；新并行n16＋final/deferred为158.52秒，约 **5.91倍吞吐**。两组均包含仿真启动、reset及清理，排除共用模型加载；单次短测不能外推所有task。

新配置4800请求长测为367.91秒：**全仿真窗口平均71.07%，首个真实请求至最后回合完成平均89.37%**。这是已测窗口，不是连续3小时、8卡整作业或任意新机器保证。详细对照、逐轨迹/RNG验证及实现边界见 [技术与性能说明](PARALLEL_SEEDED_EVALUATION.zh-CN.md)。

最终公开入口全50任务验收结果待完成后填入本节。远端535.129.03驱动尚未直接验证；这里不宣称已经定位旧r13的精确原生退出根因。

历史单进程结果见 `provenance/SEED_ONLY_EVALUATION_VALIDATION.json`；旧PLANS说明见 [历史协议](ASTRA_RECORDED_PLANS_LEGACY.zh-CN.md)。这些记录保留，不能用历史84%–85%等数字替代本入口的实际采样。
