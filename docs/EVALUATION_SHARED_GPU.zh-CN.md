# 独立任务共用 GPU：保持原单环境评测协议

`--tasks-per-gpu 2` 让同一张 GPU 同时服务两个独立任务。每个任务仍有自己的策略进程、模拟器、端口和结果，保持 batch1、n_envs=1、seed7、任务内回合顺序、原动作精度、10次去噪和每5步重规划。它通过重叠不同任务的仿真和推理提高吞吐，不对一个回合的时间步并行，也不将多个输入合成真实 batch。

每个策略副本都占显存和CPU。默认 `--tasks-per-gpu 1`、策略/模拟器OMP/MKL各4、OpenBLAS各1，原入口行为保留。新选项是资源调度设置，不代表任何硬件都能获得相同收益。

## 使用

沿用 HANDOFF 中已配置的 `PY/H/V/CFG/CKPT/SIM_PY`。先使用新的结果目录；以下只设置测试过的并发数2，线程数以本页实测结论为准：

```bash
OUT="$H/eval/${V}-70k-shared2-graph"
COMMON=(--variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT"
        --sim-python "$SIM_PY" --output "$OUT"
        --inference-backend cuda-graph --tasks-per-gpu 2
        --policy-cpu-threads 4 --sim-cpu-threads 4)

# 仅需一片smoke。该模式在同一张物理卡上同时跑两个任务、各两回合。
"$PY" scripts/evaluate.py "${COMMON[@]}" --part 1 --gpu-ids 0 --egl-device-ids 0 --smoke

# 四任务分片：两张GPU、每张两任务，最多四路同时运行。
PART=1 # 普通分片为1..12；需分别执行全部，再执行part13。不要把12片同时挤在同两张卡上。
"$PY" scripts/evaluate.py "${COMMON[@]}" --part "$PART" --gpu-ids 0,1 --egl-device-ids 0,1
# 最后两任务分片：一张GPU、两路同时运行。
"$PY" scripts/evaluate.py "${COMMON[@]}" --part 13 --gpu-ids 0 --egl-device-ids 0

# 全部50任务各50回合完成后汇总。
"$PY" scripts/evaluate.py "${COMMON[@]}" --summarize --gpu-ids 0 --egl-device-ids 0
```

如果只有一张H100，也可以对普通四任务分片用 `--gpu-ids 0 --egl-device-ids 0`：两路同时运行，每路完成本任务后再执行队列中的下一任务。不要同时启动多个分片挤在同一张卡上，否则总并发超过这里的上限；共享卡模式不自动跨进程分配资源。

CUDA/EGL设备编号应按集群内实际UUID/PCI映射填写，不能假设所有重映射环境都是0/1。`--dry-run`会列出每路GPU、任务和端口。多个变体使用各自output和不同port-base。

共享smoke每次创建新attempt，避免把历史部分结果拼成通过；要求所有模拟器的实际rollout时间存在共同重叠。CUDA Graph smoke对每次请求与eager进行完整动作逐位比较，因此smoke速度不代表正常正式模式。正式模式只在每个新策略进程首次请求检查，其余请求正常执行graph。不能通过关闭检查来掩盖动作不一致。

并发数、两类CPU线程数会进入结果identity。smoke、正式分片和汇总必须一致；换设置请使用新output并重新smoke，不能混用旧gate或结果。实际GPU数量不进identity，因此四任务与最后两任务分片可以汇总。

入口也支持 `--tasks-per-gpu 4`，但未凭2路数据宣称4路更快；需要足够显存/CPU及对应4路smoke。4路smoke要选择有四任务的分片，不能用part13。没有测试三小时低利用率回收窗口，也不保证其它权重/变体的速度或完整轨迹；必须先使用它们自己的匹配smoke。

## H100实际闭环数据

baseline70k、H10080GB、两个任务各2完整回合。同卡两个独立worker重复两次；每路仍原单环境协议。

| 设置 | 四回合rollout秒 | rollout GPU-util | 稳态GPU-util（并行取共同区间） | 相对吞吐 |
|---|---:|---:|---:|---:|
| serial_t4 | 309.95 | 18.56% | 18.98% | 1.000× |
| serial_t1 | 320.26 | 16.32% | 17.02% | 0.968× |
| parallel2_a | 181.98 | 29.30% | 32.42% | 1.703× |
| parallel2_b | 179.42 | 28.57% | 31.54% | 1.727× |

同样4回合吞吐提高1.70–1.73倍。含模拟器初始化、不含策略加载的阶段耗时379.29→217.07/214.44秒。两路峰值显存20,782MiB。

性能测试使用预先加载并公平预热的常驻server，包含运行中reset和尾部清理；不是从命令启动到50任务全部完成的耗时。两路均稳态是共同窗口，整段rollout利用率约29%，并非已经吃满H100。样本约每秒一次，中位间隔1.0003秒、最大1.3526秒。不同任务长短造成尾部单worker运行，会拉低整段util。

稳态列：串行取两段各自去掉前5请求后的有效窗口，并行取两路上述窗口的交集。整体评测利用率以rollout列为主。

sim1候选没有显示收益，因此推荐继续4/4/1；该候选阶段还并行过后续测试的CPU准备/哈希，不以约3%的差异证明单线程必然更慢。

四阶段的每回合轨迹哈希及成功标志均相同：CloseBlenderLid0/2、CloseFridge1/2。共16回合执行只对应4个独立任务回合，不是模型成功率评估。64组并发冻结输入完整动作逐位对照通过，性能阶段另有8次首请求检查。具体数值等价仍受实际硬件、输入和变体的匹配smoke约束。

## 公开入口验证

运行时代码提交 `f0922aa0e90fdc4687b5a2c76d225848bca91251`，CPU18项和706条源文件SHA通过。直接调用公开 `evaluate.py --tasks-per-gpu 2 --inference-backend cuda-graph --smoke`，在同一张H100完成两个任务共4回合、720次完整动作逐位对照；两路rollout实际重叠286.42秒，CUDA/EGL物理UUID一致，813键完整权重严格加载，全部4条完整轨迹与串行基准逐一相同。原始文件和传输包SHA已独立复核。

该公开smoke持续双算，仅证明入口及数值检查正确，不用其利用率宣传正常评测性能。上面的性能表来自正常graph闭环对照。推荐两路配置；其它变体或权重仍需匹配smoke。详细机器可读证据见[evaluation_shared_gpu_validation.json](evaluation_shared_gpu_validation.json)。

