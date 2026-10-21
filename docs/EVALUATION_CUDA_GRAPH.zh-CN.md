# 可选 CUDA Graph 评测后端

使用 `--inference-backend cuda-graph` 可以减少 PI0.5 eager 推理中的 CPU 调度空隙。默认仍为 `eager`。该选项不改变权重、精度、10 次去噪、每请求噪声种子、50 步动作块或每 5 步重规划，也不把多个环境合并成真实 batch。

## 为什么利用率低

在同一张 RTX5090 上，对 baseline 70k 的真实观测做测量：原始连续推理平均约 0.316 秒/请求，GPU-util 约 25%。8 次带 profiler 的请求平均 421ms（profiler 有额外开销），每请求记录 13,492 个 GPU 内核事件，GPU 内核和拷贝时间区间并集约 65.7ms。大量时间花在细碎 eager 调用之间的主机侧开销。单独把去噪循环改为固定次数，实测仅快约 0.8%，不是主要收益来源。

CUDA Graph 重放既有 GPU 计算，减少逐算子提交开销。为了允许捕获，只在 graph 模式缓存固定 token 布局的 attention mask，并在 GPU 上创建时间标量；普通 eager 路径保留原行为。

真 batch4 的微基准虽达到约 3.1 倍吞吐，但完整动作块出现最大绝对差约 0.084，未作为等价实现发布。

## 已取得的工程证据

RTX5090、PyTorch 2.7.1+cu128、baseline 70k，原精度和严格确定性设置：

| 测量 | eager | CUDA Graph |
| --- | ---: | ---: |
| 相同真实输入的平均推理时间 | 0.3165 秒 | 0.1069 秒 |
| 连续推理 GPU-util 样本均值 | 24.5% | 66.1% |
| 四环境试验稳态 GPU-util | 21.9% | 51.1% |
| 四个完整回合运行时间，排除环境初始化 | 244.2 秒 | 98.2 秒 |
| 四回合总耗时，包含环境初始化 | 301.9 秒 | 157.6 秒 |
| 同上整段 GPU-util | 18.4% | 30.7% |

- 48 组冻结真实观测/种子对照，完整动作块逐位一致。
- 导航和操作两类真实闭环共 270 次 shadow 请求：每次对相同观测同时计算 eager 和 graph，动作逐位一致；仿真执行 eager 的动作。
- 性能对照每种后端均完成 4 个回合、720 次推理、3600 个控制步。两组均无成功回合，这不是 benchmark 成功率估计。
- 独立完整轨迹哈希不相同。另一次重复渲染检查发现腕部相机少量像素有 1 灰度级差异；因此动作等价使用共享输入的 shadow 检查，而不是声称整条随机仿真轨迹逐位复现。
- 四环境性能试验使用独立诊断调度器。**公开 `evaluate.py` 仍使用原来的一环境评测协议，并未发布该调度器。** 上表四环境数据不能直接当成公开入口或另一集群的实测利用率。
- 以上是短期工程结果，未验证另一集群的三小时按卡回收规则。其他权重、变体和硬件必须先通过各自的匹配 smoke。

标准发布入口也已通过真实 RTX5090 smoke：直接运行 `evaluate.py --inference-backend cuda-graph --smoke`，完成 2 个 CloseBlenderLid 回合、360 次逐请求完整动作块对照、2 条不同轨迹；CUDA/EGL UUID 匹配，完整 813 键加载，writer 锁释放。该 smoke 的两回合均失败，验收的是评测实现而非模型成功率。

私有外层采集控制器曾误把应为 2 的独立轨迹数量写成 1，产生错误 HOLD；公开 `evaluate.py` 已正确生成 smoke 通过标记。之后通过独立只读工具取回 10 份原始 JSON、逐文件和传输 SHA 校验，并按正确的两条轨迹条件重新验收；原 HOLD 保留。已验证的公开运行时代码提交为 `598fc7a`，后续提交仅补充文档和 CI 依赖。原始产物传输 SHA256 为 `dac0d0bb2294f583f4a35fb48c0f1a9d637a1a1b0d119a90af3e969ab3996f90`。

## 如何使用

按主交接文档配置 `PY`、`H`、`V`、`CFG`、`CKPT`、`SIM_PY`。先使用新的输出目录执行匹配 smoke：

```bash
"$PY" scripts/evaluate.py \
  --variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT" \
  --sim-python "$SIM_PY" --output "$H/eval/${V}-70k-cuda-graph" \
  --part 1 --gpu-ids 0 --smoke --inference-backend cuda-graph
```

smoke 内部自动加 `--cuda-graph-verify`，**每次请求**都与 eager 做完整动作块逐位对照；任何差异直接报错，不生成通过标记。这会执行两份推理，所以 smoke 的速度和利用率不是正式 graph 模式性能。

smoke 通过后，保留同一输出目录、checkpoint、配置及 backend，按原来的 12×4 卡加 1×2 卡分片命令运行，仅增加：

```bash
--inference-backend cuda-graph
```

正式模式还会核对每个新策略进程的首次请求。eager 和 graph 的 smoke/结果身份不同，不能复用另一后端的通过标记。汇总时也要带相同的 `--inference-backend cuda-graph`。

直接启动策略服务时，可以在原 `scripts/server.py` 命令末尾加 `--inference-backend cuda-graph`。单独诊断数值时再加 `--cuda-graph-verify`；不要把持续双份推理的诊断模式当作性能优化。

Graph 捕获要求输入形状固定；本入口保持 native batch1。输入形状改变或数值检查失败会明确报错，不会静默改精度、减去噪步数或生成正式结果。需要恢复旧行为时使用默认 `eager` 并匹配其独立输出目录和 smoke。
