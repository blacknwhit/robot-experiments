# EXP-007 · StreamPI 在 RoboCasa365 上的训练

从原始 JAX π0.5 预训练权重开始，按单机 8 卡/global64/70,000 步或 16 卡/global128/35,001 步全量训练。准备资源、选择 worker 数、做实体 GPU 恢复验收和启动正式训练的命令见[训练与恢复手册](TRAINING.zh-CN.md)。

方法以 StreamPI 公开提交 `ba024b2149ed94542a6c3352554f4f52f766a26d` 为准。每个训练样本包含三个时刻，间隔固定为 5 帧，所有有效帧都可作为当前时刻。在线推理在编码前按周期清空历史缓存。RoboCasa 使用三路相机、逐时刻的 16 维状态和当前时刻的 50 步动作监督，环境动作维度为 12。[原始 8 卡方法与资源说明](../../docs/ASTRA_STREAMPI_8GPU_70K.zh-CN.md)

在 22 vCPU 主机上，PyTorch DataLoader 使用 8 个持久 worker，每个真实 global128 的稳定供数时间由旧版读热后的 52.92 秒降至 3.20 秒。128 个样本的图像、提示 token、掩码、状态和动作均已与旧读取器逐元素核对。16 卡端到端训练用时待目标作业实测。[同批数据对照、测速命令和验收范围](TRAINING.zh-CN.md#3-在本机复核等价性与供数速度)

`prepare.py` 生成配置和数据验收凭证，`input_pipeline.py` 负责窗口读取和多进程组批。`run.py` 与 `checkpoints.py` 执行训练和续训，`compare_runs.py` 对比连续训练与断点恢复的完整状态。`policy.py` 提供流式推理接口。模型核心位于 `runtimes/streampi`，数据集定义位于 `runtimes/streampi_reader`。
