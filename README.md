# robot-experiments

以实验为单位管理PI0.5 / RoboCasa365的代码、配置、过程与结果。

**全局入口：[EXPERIMENTS.md 实验总览](EXPERIMENTS.md)**。先选择实验，再进入对应目录；不要把新实验流程混入旧PI0.5/FIXED文档。

| 实验 | 独立入口 |
|---|---|
| EXP-001：PI0.5基线 | [实验记录](experiments/exp001_pi05_baseline/README.zh-CN.md) |
| EXP-002：Query-FIXED | [实验记录](experiments/exp002_query_fixed/README.zh-CN.md) |
| EXP-003：3D轨迹v2两组对照 | [实验记录](experiments/exp003_trajectory_v2/README.zh-CN.md) |
| **EXP-004：本轮3D位置轨迹A/B/C** | **[资源复用、GT、训练和测试手册](experiments/exp004_trajectory_v3_abc/README.zh-CN.md)** |

已有PI0.5/FIXED集群优先复用正确的数据、原始OpenPI `pi05_base`预训练权重、环境、缓存和兼容GT。EXP-004以独立输出根管理新配置、日志、checkpoint和评测结果；从原始base微调不等于加载旧70k权重，也不是全模型随机初始化。安装和下载仅在缺少资源时进行。

公共来源和历史部署参考保留在[HANDOFF.zh-CN.md](HANDOFF.zh-CN.md)，历史分数、实验状态及证据边界统一查[总览](EXPERIMENTS.md)。新增实验使用[记录模板](experiments/TEMPLATE.zh-CN.md)。

可选评测加速：[CUDA Graph使用与实测](docs/EVALUATION_CUDA_GRAPH.zh-CN.md)。保持原动作计算，通过对应smoke后显式启用；默认eager不变。

H100闭环吞吐：[独立任务共用GPU](docs/EVALUATION_SHARED_GPU.zh-CN.md)。可选每卡两路独立评测，保持每路batch1及原回合协议，通过共享smoke后启用。

以后采用[固定episode种子直接评测](docs/ASTRA_FIXED_SCENES.zh-CN.md)：`evaluate_seeded.py`读取统一任务/episode种子定义，每回合直接初始化，不预生成场景计划、不比较参考图像/state/轨迹。单卡单task多环境流水线、严格模型加载及输入/动作有效性检查保留。只有明确复现旧协议时才使用[历史兼容入口](docs/EVALUATION_LEGACY_EXACT.zh-CN.md)。此前利用率数字属于H100有限rollout实测，不能保证所有任务或整项作业；历史分数与新协议分开。

布局：

- `experiments/expNNN_name/`：每次实验独立文档、方法与配置/证据映射；大产物在各自外部根目录。
- `runtimes/baseline/`：独立历史基线源码，不是新版关闭 loss。
- `runtimes/query_fixed/`：独立 FIXED 源码与固定教师。
- `runtimes/trajectory_v2/`：最新 Query/Embedding 3D v2，共享实现、分开配置。
- `runtimes/trajectory_v3/`：EXP-004的A/B/C共享实现。
- `configs/`：完整科学配置模板；`scripts/`：可迁移入口。
- `assets/`：冻结任务清单、训练 registry、归一化统计；没有训练数据和模型参数。
- `provenance/`：来源与发布文件 SHA；`tests/`：CPU 迁移边界检查。

现有训练 checkpoint、完整 optimizer/RNG 状态以及已生成的约 350.5 GB 3D 标签不在 GitHub 中。可从旧集群搬运自有产物，或按文档从公开输入重新生成/训练。不要把只含 safetensors 的模型导出当作完整续训。

本仓库聚合了 StarVLA、OpenPI 和少量 RoboCasa/评测适配代码；各自许可证及署名保留。见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
