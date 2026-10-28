# 多任务共享GPU、固定2500回合评测

> 开发中：本分支尚未完成新入口全50任务验收。不要把下列候选参数当成已经测得的最优值；发布前需填入实际测量与验收证据。

本入口优化的是执行调度，不改变评测集：50个task，每个task固定episode 0–49，使用仓库`assets/eval_scene_seeds_v1.json`中已发布的对应种子。总数2500，不取最先完成的回合，不随GPU数、worker数或执行顺序改seed。不生成场景计划，不比较封存图像，不按旧分数筛选结果。

每个GPU常驻一份模型，多个task共享它。独立CPU仿真进程推进原RoboCasa环境；每GPU只有一个进程持有EGL并进行绘制。CPU进程发送原MuJoCo完整绘制场景，渲染端不重新计算物理，不重新编译XML。每次策略推理仍是原batch1、原精度和CUDA Graph；50步动作预测、每5步重规划、10次去噪、原horizon及成功判定不变。

## 运行

依赖、模型配置与环境准备沿用`HANDOFF.zh-CN.md`。需要MuJoCo3.3.1、真实`receipts/sim.freeze.txt`以及正确的CUDA/EGL物理设备映射。容器创建时设置`NVIDIA_DRIVER_CAPABILITIES=all`。不要复制r12/r13私有补丁。

```bash
POLICY_PY=/absolute/path/to/policy/python
SIM_PY=/absolute/path/to/simulator/python
HOME_DIR=/absolute/path/to/robot-experiments-data
VARIANT=baseline
CONFIG=/absolute/path/to/baseline-resolved.yaml
CHECKPOINT=/absolute/path/to/trained-model.safetensors
OUT="$HOME_DIR/eval/${VARIANT}-parallel-seeded-new-output"

"$POLICY_PY" scripts/evaluate_parallel.py \
  --variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --all-tasks --gpu-ids 0,1,2,3,4,5,6,7 \
  --egl-device-ids 0,1,2,3,4,5,6,7 --port-base 24090 \
  --n-envs 8 --tasks-per-gpu 2 --simulator-cpu-threads 1 --timeout 43200
```

- `--n-envs`：**每GPU总共多少个CPU仿真worker**，不是每task的数量。候选值8；实际推荐待实测。
- `--tasks-per-gpu`：一张GPU同时领取多少个task组成一批，批内交错分配episode给worker。候选值2。当前实现完成整批后再领取下一批；它不是跨GPU逐episode工作窃取。
- GPU数量由`--gpu-ids`指定，可以1、2、4、8等；每个CUDA编号对应的EGL编号必须正确。不能把`cuda:0`逻辑编号直接当宿主EGL编号。
- 全机最多GPU数×n_envs个CPU worker，加每GPU一个渲染进程和一个模型服务。按实际CPU配额调整worker数；增加模型副本不是此入口的工作方式。
- 策略OMP/MKL默认4、OpenBLAS1；CPU worker的OMP/MKL由`--simulator-cpu-threads`控制，OpenBLAS保持1。
- `--timeout`是每批task的墙钟上限，不改变episode horizon。根据实际时限设置足够预算，不能靠截短回合提速。
- 正式命令不传`--diagnostic-episodes`、`--diagnostic-max-replans`或`--diagnostic-trace`。

只跑特定task可以把`--all-tasks`替换为`--tasks WeighIngredients,PackIdenticalLunches`；它们仍分别完成固定50个回合。每次更换模型、种子定义、源码、配置或调度参数，用新的输出目录。精确相同配置的中断运行可以重执行相同命令，完整且身份相符的task结果才会跳过；未完成task不会作为正式结果计分。

## 完成与失败

正式全套结果只有在50个task均通过身份及原始episode覆盖核验后，才生成`FINAL_ACCEPTED.json`，其trials必须是2500。每个task的原始结果在`formal/`；GPU服务日志在`services/`；每批仿真、worker日志、阶段错误和退出码在`parallel-runs/`。

保留严格权重加载、首个真实请求的eager/Graph对照、有效RGB/状态/动作检查以及固定回合覆盖检查。这些与已取消的参考图像哈希门槛不同。失败回合不能跳过、不能替换种子，也不能将诊断输出复制成正式结果。

如果在原生仿真/渲染报错，保留具体worker日志、ERROR/CLEANUP_ERROR和退出码。远端Astra负责更新并运行现成入口，修复由主对话完成。当前传输限定MuJoCo3.3.1原RGB流程；未支持的native plugin、运行时GPU纹理/mesh上传会明确失败，不能静默忽略。

## 性能口径

优先看完成同一批固定回合的总墙钟时间与GPU小时，同时记录每张GPU在真实评测过程的利用率。初始化、reset及尾部等待单独报告，不能把仅活跃片段的利用率冒充全流程或连续3小时平均值。不会用额外无效计算维持利用率。
