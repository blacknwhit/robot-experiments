# robot-experiments

以实验为单位管理PI0.5 / RoboCasa365的代码、配置、过程与结果。

**全局入口：[EXPERIMENTS.md 实验总览](EXPERIMENTS.md)**。先选择实验，再进入对应目录；不要把新实验流程混入旧PI0.5/FIXED文档。

| 实验 | 独立入口 |
|---|---|
| EXP-001：PI0.5基线 | [实验记录](experiments/exp001_pi05_baseline/README.zh-CN.md) |
| EXP-002：Query-FIXED | [实验记录](experiments/exp002_query_fixed/README.zh-CN.md) |
| EXP-003：3D轨迹v2两组对照 | [实验记录](experiments/exp003_trajectory_v2/README.zh-CN.md) |
| **EXP-004：本轮3D位置轨迹A/B/C** | **[资源复用、GT、训练和测试手册](experiments/exp004_trajectory_v3_abc/README.zh-CN.md)** |
| EXP-005：Pri4R论文方法准备版 | [独立实验及远端训练说明](experiments/exp005_pri4r_paper/README.zh-CN.md) |

已有PI0.5/FIXED集群优先复用正确的数据、原始OpenPI `pi05_base`预训练权重、环境、缓存和兼容GT。EXP-004以独立输出根管理新配置、日志、checkpoint和评测结果；从原始base微调不等于加载旧70k权重，也不是全模型随机初始化。安装和下载仅在缺少资源时进行。

公共来源和历史部署参考保留在[HANDOFF.zh-CN.md](HANDOFF.zh-CN.md)，历史分数、实验状态及证据边界统一查[总览](EXPERIMENTS.md)。新增实验使用[记录模板](experiments/TEMPLATE.zh-CN.md)。

以后采用[固定2500回合的高吞吐评测](docs/ASTRA_FIXED_SCENES.zh-CN.md)：`evaluate_parallel.py`读取统一任务/episode种子定义，每回合直接初始化，不预生成场景计划、不比较参考图像/state/轨迹。每GPU一份常驻模型，多task共享CPU并行仿真及单EGL渲染进程；严格模型加载及输入/动作有效性检查保留。只有明确复现旧协议时才使用[历史兼容入口](docs/EVALUATION_LEGACY_EXACT.zh-CN.md)。历史分数与新协议分开。

2026-10-28最终公开入口已在单H100上完成**全50任务、共100完整回合**实现验收：44.67分钟，含模型加载/初始化/收尾的GPU-util采样均值**72.04%**；首请求至最后回合为82.16%。这不是2500回合正式成绩或8卡实测。相同1280请求的前期对照由936.27秒降至158.52秒。测量窗口、采样质量与边界见[最终验收证据](provenance/PARALLEL_EVALUATION_VALIDATION.json)和[技术说明](docs/PARALLEL_SEEDED_EVALUATION.zh-CN.md)；历史单进程记录仍保留。

历史入口资料：[CUDA Graph使用与实测](docs/EVALUATION_CUDA_GRAPH.zh-CN.md)、[独立任务共用GPU](docs/EVALUATION_SHARED_GPU.zh-CN.md)。其中eager默认、完整smoke和每task独立模型属于旧入口；当前并行种子入口采用`cuda-graph-pipeline`、每卡共享一份模型，无需每权重完整smoke。

布局：

- `experiments/expNNN_name/`：每次实验独立文档、方法与配置/证据映射；大产物在各自外部根目录。
- `runtimes/baseline/`：独立历史基线源码，不是新版关闭 loss。
- `runtimes/query_fixed/`：独立 FIXED 源码与固定教师。
- `runtimes/trajectory_v2/`：最新 Query/Embedding 3D v2，共享实现、分开配置。
- `runtimes/trajectory_v3/`：EXP-004的A/B/C共享实现。
- `runtimes/pri4r_paper/`：EXP-005的原PI0.5前缀、独立辅助Embedding及论文训练配方。
- `configs/`：完整科学配置模板；`scripts/`：可迁移入口。
- `assets/`：冻结任务清单、训练 registry、归一化统计；没有训练数据和模型参数。
- `provenance/`：来源与发布文件 SHA；`tests/`：CPU 迁移边界检查。

现有训练 checkpoint、完整 optimizer/RNG 状态以及已生成的约 350.5 GB 3D 标签不在 GitHub 中。可从旧集群搬运自有产物，或按文档从公开输入重新生成/训练。不要把只含 safetensors 的模型导出当作完整续训。

本仓库聚合了 StarVLA、OpenPI 和少量 RoboCasa/评测适配代码；各自许可证及署名保留。见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
