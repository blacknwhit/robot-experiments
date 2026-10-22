# 共同场景集合的多环境评测（不同于每个模型的旧场景链）

> 新评测推荐[固定种子、只reset准备2500场景的执行文档](ASTRA_FIXED_SCENES.zh-CN.md)，无需先评测基线；新固定场景正式入口不再要求每权重完整smoke，下面旧录制流程中的smoke命令不适用于新协议的必经步骤。本页保留原先从真实串行策略运行录制计划的流程与历史工程证据。

> 若要求保留之前的评测内容，使用[历史兼容入口](EVALUATION_LEGACY_EXACT.zh-CN.md)。本页跨模型共享集合必须显式选择`--scene-protocol shared-scenes`，不能据此宣称复现其他模型的历史场景。

使用新入口 `scripts/evaluate_task.py`。一个 task 的多个环境异步向同一份模型发请求，GPU仍逐请求执行原生batch=1；CPU图像预处理和GPU执行形成流水线。它与 `evaluate.py --tasks-per-gpu 2` 的“不同task共卡”是两种设置，不能叠加解释本页结果。

## 场景和动作如何保持一致

旧串行评测并非每个回合都重新设seed。环境在执行动作期间也可能消费随机数，导致下一回合场景依赖上一回合过程。因此直接把旧入口的n_envs改成8，不能保证与单环境测同一批场景。

新流程先用基线原始串行顺序评测并录制每次reset前的环境、NumPy、Python随机状态，同时保存初始图像/语言/状态和MuJoCo状态SHA。录制的评测本身就是基线结果，不需要额外再跑一遍基线。场景计划为JSON，不需要加载外部pickle。

随后所有模型共用同一份冻结的 `PLAN_SET.json` 和各task的计划。无论n_envs=1、4还是8，都恰好评测全局episode ID 0…49一次；物理worker编号不进入策略的随机种子，仍使用原logical environment index=0和原episode/replan编号。每次reset后检查实际初始场景SHA，不一致即终止。输出按episode ID排序，不取“最快结束的50个回合”。

同一checkpoint的smoke核对单环境和并行的策略输入/实际执行动作轨迹SHA、成功数组，并对每次模型请求比较完整50步动作与原eager结果，要求逐位相同。不同checkpoint应在同一批场景上比较成功率，不要求它们产生同样动作。场景集合SHA参与smoke和正式结果身份，替换计划后不能沿用旧smoke。

**新固定场景评测不能与其他模型的历史串行分数直接混为同一批回合。** 所有待比较模型应重放同一集合；历史数字继续保留为历史结果。

## 实测边界

H10080GB，baseline70k，CloseBlenderLid；原生BF16、TF32关闭、10次去噪、每5步重规划、batch1保持。下面是实际闭环，包含回合重置，排除模型加载和环境首次构建；不含逐请求双算校验阶段。

| 同一task的环境数 | 回合数 | rollout耗时 | 平均GPU-util | 峰值显存 |
|---|---:|---:|---:|---:|
| 1，原串行Graph | 8 | 469.66秒 | 16.97% | — |
| 4，异步流水线，连续两轮分配 | 8 | 144.66秒 | 57.41% | 15,674 MiB |
| 8，异步流水线，两次 | 8 | 105.23 / 105.60秒 | 84.01 / 85.24% | 22,201 MiB |

8回合的原串行参考由前4回合及真实续接的后4回合组成，rollout合计469.66秒；发布版8环境约快4.45–4.46倍。所有回合初始场景和策略输入/实际执行动作轨迹均一致。公开入口capture2及replay8共1800次完整动作逐位对照通过；随后正式计算方式共4320请求，仅首请求执行一次对照，其他请求没有重复双算。4环境×8回合覆盖了同一个worker领取后续回合的情况。

计入环境首次构建后，8环境这两次短测为137.36 / 138.86秒，平均GPU-util **65.77 / 66.03%**，模型加载另计。不能把84–85%的rollout平均值称为整个任务的平均值。原始数值与验收范围见[验证记录](evaluation_single_task_validation.json)。Linux23项CPU测试和710项源码SHA检查通过；实际GPU代码为 `15fd76968e2d4270c74d5e2421be4e39e3ca77af`，后续发布提交仅增加文档和验收记录。

这些是一个task的有限探针，不是全50task或连续3小时保证。不同任务的reset、CPU和渲染开销不同；远端应采集真实评测期间的每卡利用率。smoke额外执行eager对照，不能拿其利用率作为正式效率。

## 操作

先按主交接文档完成策略/仿真环境、配置、权重、CUDA/EGL物理GPU映射。下面的 `POLICY_PY`、`SIM_PY`、`HOME_DIR`、`CONFIG`、`CHECKPOINT` 均为本集群绝对路径。每个task一张卡；不要在同一卡另开一份模型。CPU线程沿用策略4/4/1、每个模拟器4/4/1；本次没有以减少线程为提速来源。

### 1. 基线录制统一回合集

```bash
COMMON=(--variant baseline --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$HOME_DIR/baseline-scene-capture")

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 \
  --capture-plan --n-envs 1 --smoke

# smoke通过后，各task只执行一次正式50回合录制。可跨GPU并行不同task，端口必须独立。
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 \
  --capture-plan --n-envs 1
# 对 assets 中其余49个task执行同样的正式命令。

# 所有50task完成后，同时验收2500条结果并封存统一场景集合。
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" --capture-plan --n-envs 1 --summarize
```

计划目录是 `$HOME_DIR/baseline-scene-capture/plans`。`PLAN_SET.json`为封存身份；比较模型共用完整计划及manifest。当前入口严格要求本集群真实 `sim.freeze.txt` SHA与计划一致；freeze含editable安装路径，因此更换集群或安装目录也可能被拒绝。不要复制旧freeze冒充当前环境。建议在目标集群录制一次统一集合；跨集群复用旧集合需另行核验仿真环境兼容性，当前入口不自动放宽该检查。

若只做少量task调试，可用 `python scripts/episode_plan.py --build-set PLAN_DIR` 封存子集；正式汇总仍要求50task×50回合，8回合调试计划不能冒充正式计划。保留 `.inprogress.json` 证据；只有完整Task.json进入集合。

### 2. 一个task在一张卡上用8环境重放

```bash
# 将CONFIG/CHECKPOINT/variant换成要测的同一模型，输出使用新目录。
COMMON=(--variant baseline --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" \
  --output "$HOME_DIR/baseline-fixed-scenes-n8" \
  --scene-protocol shared-scenes --plan-dir "$HOME_DIR/baseline-scene-capture/plans" --n-envs 8)

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 --smoke

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090
# 同一配置的一片smoke通过即可开始其它task，不需要每片重复smoke。
# 对其余49个task执行同样的正式重放命令，全部完成后才能汇总。

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" --summarize
```

smoke使用8个回合。若计划录制模型与当前模型相同，使用计划中真实串行参考；若不同，先在当前模型上单环境重放同8个场景，再执行并行逐请求对照。正式50回合仅首请求核验eager/Graph，随后执行已验证的Graph，避免无效双算。保持完整回合清单；不要为了提高GPU-util删掉慢场景或改变成功条件。

两卡可同时跑两个task，四卡可同时跑四个task，各task内部n_envs=8；模型仅一份/卡。CUDA ordinal和EGL ordinal的映射需按实际UUID/PCI核实，不能把所有渲染器都固定到全局EGL0。不同GPU运行使用不同端口。同一模型/集合/n_envs使用同一个output，不同模型或n_envs使用独立output。

若CPU或显存不足，使用n_envs=4及新的output重新smoke。精度、去噪步数和重规划频率保持不变。旧 `evaluate.py` 默认行为保持，新入口默认legacy；只有显式选用shared-scenes才允许跨模型共用计划。
