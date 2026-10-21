# 保留历史评测内容：一张卡一个 task

本页是需要与之前评测公平可比时的推荐入口。保持原任务清单、每任务50回合、seed=7、原生串行reset顺序、各任务horizon、每5步重规划、10次去噪、原精度与成功判定。CUDA Graph只减少调度开销；smoke对每次请求的完整50步预测动作与eager逐位比较。不会改变采样数或删除慢场景来提高利用率。

## 先区分两种情况

**第一次评测一个checkpoint：默认n_envs=1，直接运行旧场景链。** 原环境在执行动作期间可能消耗随机数，因此下一回合场景可能依赖上一回合执行过程。直接改成多环境，或让所有模型共用基线生成的场景，不能保证复现每个模型自己的旧场景链。

新入口在这一次实际评测中顺便保存reset RNG、初始观测/物理状态SHA及结果；不是先做完整串行录制，再额外评测一次。不能把后者的重放速度当作首次评测加速。

**已有这个确切checkpoint/config的原生场景计划：可以用n_envs=4或8并行重放。** 入口拒绝其他checkpoint/config的计划，并要求正式全部50回合的输入/执行动作轨迹和成功数组与原串行参考一致。旧结果若只有轨迹SHA而没有reset RNG，无法从SHA恢复场景计划。并行处理仍保持原batch=1，由CPU预处理与GPU推理流水线提高吞吐。

此前H100单task、8环境84–85%的rollout利用率属于后一种已有场景计划的重放；计入初次环境构建约66%，模型加载另计。它不是任意新checkpoint首次评测的保证，也不是连续3小时或全部50任务的保证。

## 环境与权重准备

先完成[交接文档](../HANDOFF.zh-CN.md)中的策略/仿真环境、数据资产、权重和配置准备。使用当前环境真实生成的`receipts/sim.freeze.txt`，不要复制其他集群的freeze冒充本机环境。代码、配置、模型、仿真freeze及场景集合都参与验收身份。

确认CUDA和EGL ordinal实际指向同一GPU UUID/PCI地址。策略使用`CUDA_VISIBLE_DEVICES`，模拟器使用自己的EGL设备号，不默认两种编号在容器重映射下相等。每张卡只开一个task；不同卡用不同端口。默认策略和模拟器线程均沿用已测试的4/4/1（OMP/MKL/OpenBLAS），不要同时在同一卡叠加多个launcher。

下例变量均为本集群绝对路径；`VARIANT`是`baseline`、`query_fixed`、`query3d_v2`或`embedding3d_v2`，必须与配置一致。

## 首次评测，保留旧内容

```bash
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" \
  --output "$HOME_DIR/eval-legacy-$VARIANT" --scene-protocol legacy --n-envs 1)

# 一个任务的应用smoke通过即可，不要求50个任务全部先smoke。
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 --smoke

# 正式50回合；smoke回合不计入正式结果。
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090
```

默认无`--plan-dir`时就执行原生串行流程并同步录制，不必加`--capture-plan`。模型每task加载一次，不能把每个回合单独启动成一次任务。

对`assets/eval_tasks.json`中其余49个task执行相同正式命令。可以同时分配到2/4张卡，每卡一个task；所有task共用上述output，端口各自独立。每个task有文件锁和身份receipt，完整结果可校验后跳过；中断文件保留，不伪装成完整结果。

```bash
# 全部50task完成后才汇总，要求2500个唯一回合轨迹。
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" --summarize
```

结果是`formal/<Task>.json`及身份receipt，场景计划在`plans/`，最终`FINAL_ACCEPTED.json`记录2500回合覆盖、模型与源码身份、场景集合SHA。单个task平台退出成功不代替这些应用验收。

如果有该模型历史原始任务JSON，可以在COMMON加入`--historical-results /absolute/old-results`。支持旧`atomic_seen/`等子目录，但同一task出现两份会拒绝。每次选中回合的成功数组及轨迹必须与历史相同；少于正式50回合的参考不能通过正式验收。旧记录缺checkpoint SHA时会明确标注未验证，不能据此声称权重身份已核实。只允许带相同checkpoint SHA的旧Query/Embedding名称别名。

## 已有自己场景计划时使用多环境

```bash
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" \
  --output "$HOME_DIR/eval-legacy-replay-$VARIANT-n8" \
  --scene-protocol legacy --plan-dir "$HOME_DIR/eval-legacy-$VARIANT/plans" --n-envs 8)

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 --smoke
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090
# 其余49task完成后：
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" --summarize
```

全局episode编号固定为0…49，worker编号不进入策略seed，所有初始场景SHA逐回合检查；结果按episode编号排序。smoke检查8个回合且每请求比较完整50步动作，正式阶段只做首次eager/Graph数值检查，全部50条输入/执行动作轨迹仍需与原生参考一致。不能只凭前8回合相同而放行后42回合。

少量task调试计划可用`python scripts/episode_plan.py --build-set PLAN_DIR`封存，完整评测汇总仍要求50×50。CPU/显存不足可将n_envs改4并使用新output重新smoke。现有跨集群计划受真实sim.freeze SHA与初始场景SHA约束，不能跳过检查强行复用。

## 可选中间RGB优化与性能解释

`--rgb-last-frame`仅用于原生串行模式，需在smoke、正式和汇总命令中一致指定。它跳过每个5步动作块中前4步未被策略读取的RGB渲染，仍执行所有物理子步、观测采样时钟、奖励/成功判断和随机数操作；最后实际控制步正常渲染，短尾块同样处理。

只支持当前三RGB相机、无深度/分割/录像、单帧观测、默认无噪声/滤波/延迟的配置，条件不满足立即报错。该模式smoke先跑原生版本，再跑优化版本，要求动作轨迹、成功数组、reset RNG和初始场景一致。正式结果必须记录实际缓存/渲染计数。它不是提高利用率的保证：渲染只占部分耗时，减少GPU渲染工作甚至可能降低GPU-util，但提升有效吞吐仍是优化。

具体H100实测与公开入口验收见[evaluation_legacy_exact_validation.json](evaluation_legacy_exact_validation.json)。这里的有限smoke不等于四种模型、全部50任务或整个3小时窗口都已复测。新集群/模型必须通过自己的smoke。

最终优先看相同内容的有效回合/秒，并同时记录真实每卡util；不要使用空转或重复推理“填满”卡。若新checkpoint首次评测仍受串行CPU物理仿真限制，不能在保持旧场景链的前提下承诺70–80%util。

## 不用于本次历史对比的模式

`--scene-protocol shared-scenes`才允许跨模型使用共同计划，详见[共同场景实验](EVALUATION_SINGLE_TASK.zh-CN.md)。它可支持新实验内部的统一场景对比，但不能将其他模型自己的旧分数直接并入。默认`legacy`不允许这种混用。

旧`episode_trace_sha256`覆盖策略输入和实际执行的5步动作块；完整50步预测动作的逐位一致由独立eager/Graph校验提供，不把两者混称为同一种证据。
