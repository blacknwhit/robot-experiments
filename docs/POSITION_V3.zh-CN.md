# Embedding 位置监督 A / 时间解码 B：工程验证

2026-10-20。这两个配置是新的实验候选，**没有70k成绩，也没有证明性能改善**。当前已经开展训练、恢复、推理/评测及有限短程验证。新集群全部操作见[完整交接](POSITION_V3_HANDOFF.zh-CN.md)。

| 配置 | 母结构 | 输出与监督 | 新增解码层 |
|---|---|---|---|
| `embedding3d_pos_a` | 训练期 Embedding，不增加动作 Query | 全50步相对起点位置L1＋相邻差分L1 | 无 |
| `embedding3d_temporal_b` | 同A | 同A | 2层，同一点沿时间因果attention |

B 保留 PointMLP、4096→256 融合投影及 GELU，在256→3输出层之前加入两个 pre-LN 残差块。每层8头、FFN 512、GELU、dropout=0；不增加RoPE或另一套位置编码。各点共享参数且分别处理长度50序列；后时刻可以读取同一点更早时刻与自身的隐藏特征。它不是反馈预测XYZ的自回归，也不是跨点attention。上游Embedding仍有双向时间槽交互，所以因果声明仅针对新增解码层。

数据仍是同一批机械臂/直接物体GT，全部坐标固定在窗口起始机器人坐标系。新reader直接用实际位置构造相对起点目标；完整有效前缀以外不监督，不跨越缺帧、不把末尾补零当静止真值。真实静止点保留。未来GT及其mask仅用于loss，不输入模型。

学习率日程仍为100k时域，global64/acc1、seed42、G8和相机角色增强沿用v2。当前辅助系数1.0仅用于工程调试，尚未完成新目标系数校准；这是未经效果校准的候选系数。入口默认仍限制smoke最多10步、短验证最多1500步；仅显式`--formal`生成最多70k的新候选配置，不自动启动任务。

```bash
python scripts/configure.py --variant embedding3d_pos_a \
  --home "$ROBOT_EXPERIMENTS_HOME" --run-id pos_a_smoke --gpus 2 \
  --smoke --stop 2 --out resolved/a-smoke-2.yaml
python scripts/train.py --config resolved/a-smoke-2.yaml
# 生成相同run-id、仅stop改为3的配置，在新进程中 --resume。
python scripts/configure.py --variant embedding3d_pos_a \
  --home "$ROBOT_EXPERIMENTS_HOME" --run-id pos_a_smoke --gpus 2 \
  --smoke --stop 3 --out resolved/a-smoke-3.yaml
python scripts/train.py --config resolved/a-smoke-3.yaml --resume
```

B使用`embedding3d_temporal_b`及独立run-id。短程验证使用`--engineering --stop 1500`与独立run-id，从相同base开始，不使用smoke权重。数据、环境和真实GT验收仍需先按主交接文档完成。旧v2的同形状输出不兼容新位置语义；旧权重与A/B互换恢复都会被签名拒绝。

测试命令：

```bash
python -m unittest discover -s tests -v
```

CPU检查覆盖目标/梯度的独立公式、固定坐标、有效前缀、未来阻断、点置换与分块、共同模块初始化及CPU Adam恢复。这不替代真实GPU/NCCL、完整ZeRO状态、真实样本动作推理与EGL闭环smoke。

GPU smoke会记录各rank共享VLM、Embedding各层、点MLP和新增时间层的真实梯度与FP32主权重更新；第1步额外完整保存，恢复使用完整模型、Adam、scheduler、RNG及数据进度。严格动作推理拒绝执行点头，不需要3D输入。

高效评测统一使用`scripts/evaluate_frozen.py`，共享固定场景、单卡单task、多环境异步及CUDA Graph流水线；普通正式评测保留首请求完整动作一致性门槛，不要求每套权重复做完整smoke。开发者可显式`--smoke`对照单环境/多环境，见专用交接。旧`evaluate.py`保留历史协议，短测结果不等于正式成绩。
