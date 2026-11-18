# StreamPI 在 RoboCasa365 上的训练与恢复

如果要用 16 张卡跑新版，选择 **单机 16 GPU、global batch 128、8 个 DataLoader worker、35,001 次更新**。一个 Python 进程管理整台机器，8 个 worker 是整台机器的数据读取进程总数。两台各 8 GPU 的部署需要多主机训练入口，当前 `run.py` 只接受一台机器上对同一进程可见的全部设备。

| 配方文件 | GPU | 每卡 batch | Global batch | FSDP 组大小 | 停止步 | 示例 worker 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `recipe_8gpu64_70k.json` | 8 | 8 | 64 | 8 | 70,000 | 4 |
| `recipe_16gpu128_35001.json` | 16 | 8 | 128 | 8 | 35,001 | 8 |

16 卡示例保留 8 卡分片组，在单机上形成两组。若目标部署使用 FSDP16，可另存配方，将 `fsdp_devices` 设为 16，再在该拓扑上做第 4 节验收。两份配方均从 JAX `pi05_base/params` 开始全量训练，梯度累积为 1，warmup 10,000 次更新，沿用官方 1M 步的学习率日程和 EMA 0.999。35,001 是此次 16 卡配置的停止步，比 70,000 × 64 的总样本数多一批 128 个样本。

## 1. 准备代码和已有资源

下面的 Bash 命令在同一个 shell 中执行。将前两个变量改为授权工作区和已有的兼容 Python 路径。第一次使用新版时创建独立 checkout。已有任务的目录与断点见第 6 节。

```bash
set -euo pipefail
export ROBOT_WORKSPACE=/YOUR_AUTHORIZED_ROOT
export POLICY_PY="$ROBOT_WORKSPACE/envs/existing-jax/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CODE_DIR="$ROBOT_WORKSPACE/robot-experiments-streampi-loader"
git clone --branch main https://github.com/blacknwhit/robot-experiments.git "$CODE_DIR"
cd "$CODE_DIR"
REV=$(git rev-parse HEAD)
git switch --detach "$REV"
EXP="$CODE_DIR/experiments/exp007_streampi"
RESOURCES="$ROBOT_WORKSPACE/streampi-resources.json"
if [[ ! -f "$RESOURCES" ]]; then cp "$EXP/resources.example.json" "$RESOURCES"; fi
```

在本机的 `$RESOURCES` 填写这六个已有路径。所有路径都应在授权工作区内。依赖版本与数据协议详见 [π0.5 资源与方法说明](../../docs/ASTRA_STREAMPI_8GPU_70K.zh-CN.md)。

| 键 | 填写内容 |
| --- | --- |
| `base_checkpoint` | 原始官方 JAX π0.5 的 `params` 目录 |
| `norm_assets` | RoboCasa 状态和动作的归一化统计目录 |
| `tokenizer` | 本地 PaliGemma tokenizer 文件 |
| `data_root` | 同时含 `robocasa365-pretrain-atomic` 与 `robocasa365-pretrain-composite` 的目录 |
| `registry_file` | 当前数据版本的 RoboCasa `dataset_registry.py` |
| `arrow_cache` | 与这两份 LeRobot v3 数据匹配的已准备 Arrow 缓存 |

## 2. 选择配方并生成配置

默认示例为 16 卡。复现 8 卡原配方时，将第一行改为 `GPUS=8`。在此处一次选好 worker 数，短训和正式配置都使用这个值。每组新的验收使用独立目录。

```bash
GPUS=16
case "$GPUS" in
  16) RECIPE="$EXP/recipe_16gpu128_35001.json"; WORKERS=8 ;;
   8) RECIPE="$EXP/recipe_8gpu64_70k.json"; WORKERS=4 ;;
   *) echo "Choose 8 or 16 GPUs" >&2; exit 2 ;;
esac
TAG="zf-streampi-g${GPUS}-$(date -u +%Y%m%dT%H%M%SZ)"
CFGDIR="$ROBOT_WORKSPACE/$TAG-configs"
mkdir "$CFGDIR"
printf '%s\n' "$REV" > "$CFGDIR/source.commit"
REF="$TAG-ref"
RESUME="$TAG-resume"
FORMAL="$TAG-train"
for ID in "$REF" "$RESUME"; do
  "$POLICY_PY" "$EXP/prepare.py" configure \
    --workspace-root "$ROBOT_WORKSPACE" --resources "$RESOURCES" \
    --recipe "$RECIPE" --loader-workers "$WORKERS" \
    --run-id "$ID" --mode smoke --out "$CFGDIR/$ID.json"
  "$POLICY_PY" "$EXP/prepare.py" check-data --config "$CFGDIR/$ID.json"
done
bash "$EXP/launch.sh" --config "$CFGDIR/$REF.json" --dry-run
```

`configure` 封存源码、资源和采样配置。`check-data` 检查 300 个任务、30,000 条演示，对 4 个真实样本逐一比较新旧读取器，并通过真实 DataLoader 组出完整 global batch。每个运行目录各有一份 `DATA_ACCEPTED.json`。对 16 卡示例，核对 `real_global_batch=128`，每个相机的图像形状为 `[128,3,224,224,3]`，提示 token 为 `[128,3,200]`，动作为 `[128,50,32]`。

## 3. 在本机复核等价性与供数速度

这两项独立诊断只使用 CPU，可在分配 GPU 前执行。第一条把给定索引范围内的所有模型输入与旧读取器逐元素比较。第二条用相同的索引顺序重复取四批，将首批启动与后三批稳定耗时分别输出。

```bash
"$POLICY_PY" "$EXP/tools/check_loader.py" compare \
  --config "$CFGDIR/$REF.json" --start 0 --batches 1 --workers "$WORKERS" | tee "$CFGDIR/input-compare.log"
"$POLICY_PY" "$EXP/tools/check_loader.py" bench \
  --config "$CFGDIR/$REF.json" --start 0 --batches 4 --workers legacy 0 4 8 legacy | tee "$CFGDIR/reader-bench.log"
```

对照通过会输出 `exact_all_model_inputs=true`。将 `--start` 改为要验证的全局样本游标即可检查非零起点。`legacy` 调用旧逐样本读取器，`0` 在训练进程内使用新窗口读取器，正数使用相应数量的持久 worker。测速临界区覆盖解码、进程间传递和组批，输入校验独立执行。每相机一次打开视频取三个历史时刻，H50 动作标签只在当前时刻构造。

在配额为 22 vCPU 的 H100 主机上，同一组 512 个真实样本测得下列结果。每行 4 批，各 128 个样本。

| 读取方式 | 首批（秒） | 后三批均值（秒/批） |
| --- | ---: | ---: |
| 旧逐样本读取，第一轮 | 88.18 | 53.59 |
| 新窗口读取，0 worker | 24.79 | 25.64 |
| 新窗口读取，4 worker | 117.94 | 6.34 |
| 新窗口读取，8 worker | 178.76 | 3.20 |
| 旧逐样本读取，最后复测 | 51.56 | 52.92 |

8 个 worker 对最后一轮旧版的稳定供数吞吐为 **16.54 倍**。完整 global128 的全部输入逐元素一致。旧版在每一步对三路 FP32 图像复制并计算 SHA256，global128 为 661.5 MiB，主机 CPU 同形状缓冲区实测中位 0.679 秒。新版训练循环移除了这项工作。实体 16 卡的总步时、吞吐和利用率须以正式作业的实测为准。

## 4. 目标 GPU 上验收短训和恢复

让单个 JAX 进程看到本配方的全部 GPU。在同一组设备上顺序运行下面三个进程。先用一条参考轨迹连续更新到第 3 步，再让另一条轨迹在第 2 步保存、退出并恢复到第 3 步。

```bash
bash "$EXP/launch.sh" --config "$CFGDIR/$REF.json" --stop-after 3
bash "$EXP/launch.sh" --config "$CFGDIR/$RESUME.json" --stop-after 2
bash "$EXP/launch.sh" --config "$CFGDIR/$RESUME.json" --stop-after 3
GATE="$CFGDIR/GPU${GPUS}_ACCEPTED.json"
"$POLICY_PY" "$EXP/compare_runs.py" \
  --reference "$ROBOT_WORKSPACE/$REF" --resumed "$ROBOT_WORKSPACE/$RESUME" --out "$GATE"
```

验收凭证应有 `passed=true`、`exact_full_state=true` 和对应的 `physical_gpu_count`。核对恢复目录的 `attempts.jsonl` 中 `start_step=2` 及 `restored_from`。此时 global128 的 `next_sample` 为 256，global64 为 128。比较器核对第 3 步的参数、Adam、EMA、RNG 和游标。

已有的单 H100 诊断使用 GPU 前反向与 CPU AdamW/EMA，真实样本第一步 loss 为 0.338175118，与改动前一致。让全套状态驻单张 H100 时，按需 BFC 和 96% 预分配均在申请 15.83 GiB 临时缓冲区时 OOM。8 个虚拟 CPU 设备的小模型已通过精简指标与详细指标的恢复一致性检查。这里的实体 8/16 卡短训与恢复仍由目标部署执行。

## 5. 正式训练与计时

使用同一配方、worker 数及第 4 节得到的凭证生成正式配置。正式训练在新 run 中从预训练基座开始。

```bash
"$POLICY_PY" "$EXP/prepare.py" configure \
  --workspace-root "$ROBOT_WORKSPACE" --resources "$RESOURCES" \
  --recipe "$RECIPE" --loader-workers "$WORKERS" --gpu-acceptance "$GATE" \
  --run-id "$FORMAL" --mode formal --out "$CFGDIR/$FORMAL.json"
"$POLICY_PY" "$EXP/prepare.py" check-data --config "$CFGDIR/$FORMAL.json"
bash "$EXP/launch.sh" --config "$CFGDIR/$FORMAL.json" --dry-run
bash "$EXP/launch.sh" --config "$CFGDIR/$FORMAL.json"
```

最后一行是分配到目标 GPU 后的作业启动命令。通过 SCO 时，使用单机对应卡数的 Worker Spec、`zf-` 任务名、标准优先的配额顺序和原生十分钟自动续训。将最后一行的路径展开为绝对路径，并在作业中设置同一个 `POLICY_PY`。记录配置位置、提交 SHA、任务 ID、原生 monitor ID 和 rootJob/currentJob。每次启动一个训练进程。

正式 run 的 `attempts.jsonl` 记录实际设备、global batch、worker 数及恢复游标。`metrics.jsonl` 每次更新写一行。对比完整步时看 `wall_s`，单列首批、首编译和 checkpoint 步。定位供数看 `loader_wait_s` 与 `host_pack_s`，GPU 更新及等待中的传输体现在 `seconds`，保存耗时见 `checkpoint_s`。`h2d_dispatch_s` 只计异步传输的提交时间。旧版本的 `seconds` 从取数和哈希之后开始计时。

到达停止点时检查 `FINISHED.json`、对应的完整 checkpoint 及 `exports/step_*/COMPLETE.json`。16 卡配方的最终 `step=35001`、`next_sample=4480128`，8 卡配方为 `step=70000`、`next_sample=4480000`。

## 6. 已经在跑的任务

- **旧版在跑或需要续训**。保留启动时的 checkout、resolved config、run 目录及原启动命令。该版本可以用兼容的完整 checkpoint 继续原有进度。
- **已用新版开始，作业中断**。重新执行第 5 节最后一行，指向原来的正式配置。入口在原 run 中寻找最新的完整兼容 checkpoint，恢复模型、Adam、EMA、RNG、步数和数据游标。重启后从 `attempts.jsonl` 核对接续位置。
- **旧版断点希望切换新版**。本次交付提供新 run 的训练入口和同版本恢复。旧断点带有旧配置与源码身份，直接替换源码或改 worker 数会触发身份检查。采用新版可以独立开 run。若必须保留旧步数，跨版本完整状态迁移还需在旧断点副本上验证连续更新的一致性。多机断点也需单独验证。
