# C：Query50直接Z_t＋B时间点头

2026-10-24，按用户指定增加第三组。配置名`query3d_temporal_c`，代码仍使用`runtimes/trajectory_v3`。本次范围是模型结构修改和测试/评测入口检查，不启动SCO任务、不重跑全部GPU验收。

## 与B的精确差异

```text
B：观测prefix → VLM → 独立Pri4R Embedding → Z_t → B Point Track Head
C：观测prefix + 50个可学习Query → VLM → 取末50个attention后特征 = Z_t
                                                       ↓
                                              同一个B Point Track Head
```

每个Query对应一个预测时刻，取主干最终归一化后的`[B,50,2048]`特征，直接送到点头；没有额外Embedding、投影或detach。点损失更新Query与共享VLM。动作头仍读取包含Query的prefix，所以动作损失也可以更新Query。

沿用旧Query50的`x_to_q_then_action`：观测token读取观测，Query读取观测和全部Query，动作读取观测、全部Query和动作块。不是把观测与Query改成完全双向attention。主干原有RoPE保留；点头不额外加RoPE。

点头通过共享实现保持与B一致：当前点PointMLP、4096→256融合＋GELU、两层同一点沿时间的因果attention、256→3输出。时间块width256/8heads/FFN512/pre-LN/残差/dropout0；每一点只读取自己的过去和当前时间特征，点之间不attention。上游50个Query可彼此读取，因此新增点头的因果性不代表整个Z_t生成过程严格时间因果。

相对B的YAML仅改两项：

```yaml
framework.pi05.foresight.enabled: true
framework.pi05.point_track.version: pri4r_query_zt_v1
```

完整配置在`configs/query3d_temporal_c.yaml`。GT、完整50步相对起点位置L1＋相邻差分L1、辅助系数1.0及1000步ramp、G8、相机增强、seed42、global64/acc1、AdamW、100k时域LR和70k停止设置均保持B不变。配置中保留的`embedding_layers`等字段对C不生效，不会构造Embedding。

## 训练和评测入口

环境、下载、GT只需沿用[完整交接](POSITION_V3_HANDOFF.zh-CN.md)。生成C配置示例（只生成配置，不自动训练）：

```bash
V=query3d_temporal_c
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-short-s42" \
  --gpus 2 --engineering --stop 1500 --out "resolved/${V}-short.yaml"
```

训练仍调用`train_or_resume.py`；正式候选仍须显式生成`--formal`配置。C使用独立run-id和输出，不能把A/B或旧Query v2权重当成C的完整续训状态；点头语义签名和完整配置校验会拒绝不兼容恢复。

评测继续用`evaluate_frozen.py --variant query3d_temporal_c`，其余参数按原交接填写：同一固定场景、单卡一个task、4卡/2卡分片、多环境异步、原CUDA Graph流水线。完整C权重严格加载，动作推理保留50个Query的KV缓存，但禁止调用Point Track Head；不需要3D输入。没有为C复制一套评测器或改变回合协议。

训练smoke的审计已识别C：记录`future_queries`、共享VLM、点MLP/fusion和两层时间块的梯度及FP32更新，不再访问不存在的Embedding。`point_inference_probe.py`和策略server检查实际结构与variant相符。

## 本次检查范围

```bash
python -m unittest discover -s tests -p test_query_position_v3.py -v
```

7项定向CPU检查通过：配置只改结构、B/C共享decoder输出/梯度相同、真实forward接线选取attention后的Query且梯度连通、旧Query mask和动作KV路径、同点时间因果性、checkpoint签名/运行时结构约束，以及4卡/2卡评测分发。forward和sampler测试运行生产方法，主干用轻量attention夹具；没有加载完整PI0.5权重。

另通过2项既有A/B相邻回归检查（共同初始化/分块梯度、时间因果性），并与修改前版本比较A/B初始化状态、RNG和输出。没有重跑全量训练、NCCL、真实GPU/CUDA Graph或模拟器benchmark；**C尚无GPU训练/闭环评测实测结论和效果成绩**。历史A/B验收留存其原发布身份，本次代码更新不改变运行中的封存release。
