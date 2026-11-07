# EXP-007：StreamPI → RoboCasa365

主交接入口：[Astra 8卡/global64/70k训练与恢复手册](../../docs/ASTRA_STREAMPI_8GPU_70K.zh-CN.md)。

当前协议：固定官方源码 `ba024b2149ed94542a6c3352554f4f52f766a26d`，论文/代码冲突时以代码为准。T3、固定5帧间隔、所有有效帧可作训练anchor、编码前周期flush，RoboCasa三相机/16D state/12D action/H50保持既有适配。从原始JAX π0.5全量微调；8卡单机JAX FSDP、每卡8/acc1/global64，停止70000步，官方warmup10000/peak5e-5/1M日程与EMA.999不变。

代码：`prepare.py`解析资源并核对真实数据；`run.py`正式训练与自动恢复；`checkpoints.py`全state原子保存/恢复；`compare_runs.py`实体8GPU恢复验收；`policy.py`流式推理接口；`test_cpu.py`CPU8虚拟设备检查。模型核心在`runtimes/streampi`，RoboCasa读取器在`runtimes/streampi_reader`。

本发布没有启动正式训练。实体8卡显存/通信/恢复验收待Astra完成，正式配置要求相同配方的GPU验收凭证。历史单H100结果、随机采样版本与本入口的验证范围在主手册和provenance中分开记录。
