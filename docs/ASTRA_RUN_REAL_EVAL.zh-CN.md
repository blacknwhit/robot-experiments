# 给远端 Astra：直接执行真实全量评测

> 用户已改为“以后采用新的固定场景，不复原历史场景、不为此重测旧结果”。最新执行文档是 [ASTRA_FIXED_SCENES.zh-CN.md](ASTRA_FIXED_SCENES.zh-CN.md)。以下只保留给明确需要旧串行场景链的使用者，不能用于承诺首次八环境高利用率。

请使用本集群已有的环境、当前待测权重及对应配置，更新代码后直接完成真实全量评测。无需重新实现修复、开展性能诊断或另外设计实验。

## 1. 更新代码并确定本地路径

在仓库的 `main` 分支执行，保留已有本地配置和结果：

```bash
git pull --ff-only origin main
```

使用现有实际路径填写下面变量；配置中的 variant 必须与待测模型一致。每套权重使用独立结果目录。

```bash
POLICY_PY=/absolute/path/to/policy/python
SIM_PY=/absolute/path/to/simulator/python
HOME_DIR=/absolute/path/to/robot-experiments-data
VARIANT=baseline  # 按实际模型选 baseline/query_fixed/query3d_v2/embedding3d_v2
CONFIG=/absolute/path/to/resolved-config.yaml
CHECKPOINT=/absolute/path/to/model.safetensors
OUT=/absolute/path/to/new-evaluation-output
```

GPU与EGL设备号使用本集群已确认对应同一物理卡的映射；不同卡使用不同端口。

## 2. 只做入口必需的一次 smoke

本次执行首次真实评测：一张卡一个task，使用 `legacy`、`n_envs=1` 和中间RGB优化。保持原任务、场景生成流程、seed、horizon、精度、10次去噪、每5步重规划和成功条件。

```bash
COMMON=(--variant "$VARIANT" --home "$HOME_DIR" --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY" --output "$OUT" \
  --scene-protocol legacy --n-envs 1 --rgb-last-frame)

"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090 --smoke
```

示例中的设备号0按实际映射替换。当前权重/配置的 smoke 通过后立即开始全量；不需要50个任务分别重复 smoke。若入口报错，保留日志并汇报，不绕过验收或自行改模型代码。

## 3. 执行全部50个任务，每任务50回合

读取 `assets/eval_tasks.json` 的完整任务清单，为每个task执行下列正式命令；入口自动运行50回合，smoke回合不计入正式结果。

```bash
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" \
  --task CloseBlenderLid --gpu-id 0 --egl-device-id 0 --port 19090
```

将 `CloseBlenderLid` 依次替换为清单中全部50个task。有多张卡时，在各卡上并行不同task，每卡同时只运行一个task；同一套权重的所有任务共用上述 `OUT`，每张卡使用独立端口。一个task结束后再在该卡运行下一个，不把每个回合单独启动成任务。

无需先额外录制一遍完整评测。这个正式过程本身会同步保存场景记录。

## 4. 全部完成后汇总并报告

```bash
"$POLICY_PY" scripts/evaluate_task.py "${COMMON[@]}" --summarize
```

以 `FINAL_ACCEPTED.json` 完成为准，保留 `formal/` 中的原始结果和身份记录。最终报告：测试的模型/权重、代码版本、总成功数及成功率（2500回合）、逐任务结果、总耗时和结果目录。

详细参数说明见 [EVALUATION_LEGACY_EXACT.zh-CN.md](EVALUATION_LEGACY_EXACT.zh-CN.md)。本页就是本次执行指令，不需要另外选择共同场景协议或重新进行优化研究。
