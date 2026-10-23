# robot-experiments

PI0.5 / RoboCasa365 的可迁移实验代码：原始基线、最佳 Query-FIXED、3D轨迹辅助监督v2，以及新的位置监督A、时间解码B和Query50时间解码C。

**完整操作文档：[新集群交接文档](HANDOFF.zh-CN.md)**。从公开数据与干净基座开始，包含环境、数据筛选、3D 标签生成、保存/恢复 smoke、70k 训练、4/2 GPU 正式评测和原始结果验收。

**运行最新三组实验，从[统一A/B/C交接手册](docs/POSITION_V3_HANDOFF.zh-CN.md)开始：先生成并核验一套GT点轨迹 → A/B/C各自从base训练到70k → 使用同一固定2500场景、4卡/2卡单卡单task评测 → 汇总三个结果。** 手册包含环境/数据下载、GT的8个CPU分片、三份独立配置与权重路径、续训和39个评测分片的运行命令。三组共用独立`runtimes/trajectory_v3/`；B/C点头不额外加RoPE。发布文档本身不启动任务，现有验收边界见手册末节。
**远端 Astra 新固定场景评测：[执行指令](docs/ASTRA_FIXED_SCENES.zh-CN.md)**。一次性只初始化固定2500场景，之后同集合上一卡一个task、多环境异步评测；无需先跑基线录制，旧分数单列。

2026-10-24增加`query3d_temporal_c`：50个Query参与VLM prefix，attention后的Query直接作为`Z_t`，Point Track Head与B共用同一实现；GT、监督和训练配方保持B不变。[C结构与使用范围](docs/QUERY50_TEMPORAL_C.zh-CN.md)。本次只做结构及推理接线的针对性CPU检查，没有启动训练或重复全量GPU/评测验收。

| 配置 | 方法 | 原实验 70k 结果，50 tasks × 50 trials |
|---|---|---:|
| `baseline` | 原 PI0.5，无 Query/辅助分支 | 391/2500，15.64% |
| `query_fixed` | G8、64 Query、固定初始教师的未来视觉表征监督 | **559/2500，22.36%** |
| `query3d_v2` | G8、50 Query、相邻 + 累计 3D 轨迹损失 | 373/2500，14.92% |
| `embedding3d_v2` | 原 prefix + 训练期 Embedding、相邻 + 累计 3D 轨迹损失 | 443/2500，17.72% |

以上为旧集群已经完成的历史实验结果，**不是本公开包在新集群重新跑出的分数**。最佳 FIXED 指 70k，不是 100k。新版 3D 两版没有超过该 FIXED；这里忠实保存实验，不宣称性能提升。

```bash
git clone https://github.com/blacknwhit/robot-experiments.git
cd robot-experiments
export ROBOT_EXPERIMENTS_HOME=/your/absolute/shared/robot-experiments-data
# 按 HANDOFF.zh-CN.md 顺序准备环境、数据、基座、smoke，然后训练/评测。
```

可选评测加速：[CUDA Graph使用与实测](docs/EVALUATION_CUDA_GRAPH.zh-CN.md)。保持原动作计算，通过对应smoke后显式启用；默认eager不变。

H100闭环吞吐：[独立任务共用GPU](docs/EVALUATION_SHARED_GPU.zh-CN.md)。可选每卡两路独立评测，保持每路batch1及原回合协议，通过共享smoke后启用。

以后采用[新的固定2500场景协议](docs/ASTRA_FIXED_SCENES.zh-CN.md)：一次性只reset准备场景，后续模型共享同一集合并使用单卡单task多环境流水线。只有明确需要各模型旧串行场景链时，才使用[历史兼容入口](docs/EVALUATION_LEGACY_EXACT.zh-CN.md)。此前84–85%是H100单task有限rollout实测，不能保证所有任务或整项作业达到同样利用率；历史分数不混入新集合。

布局：

- `runtimes/baseline/`：独立历史基线源码，不是新版关闭 loss。
- `runtimes/query_fixed/`：独立 FIXED 源码与固定教师。
- `runtimes/trajectory_v2/`：最新 Query/Embedding 3D v2，共享实现、分开配置。
- `configs/`：完整科学配置模板；`scripts/`：可迁移入口。
- `assets/`：冻结任务清单、训练 registry、归一化统计；没有训练数据和模型参数。
- `provenance/`：来源与发布文件 SHA；`tests/`：CPU 迁移边界检查。

现有训练 checkpoint、完整 optimizer/RNG 状态以及已生成的约 350.5 GB 3D 标签不在 GitHub 中。可从旧集群搬运自有产物，或按文档从公开输入重新生成/训练。不要把只含 safetensors 的模型导出当作完整续训。

本仓库聚合了 StarVLA、OpenPI 和少量 RoboCasa/评测适配代码；各自许可证及署名保留。见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
