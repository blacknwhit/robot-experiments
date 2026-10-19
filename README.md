# robot-experiments

PI0.5 / RoboCasa365 的可迁移实验代码：原始基线、最佳 Query-FIXED，以及最新 3D 轨迹辅助监督的 Query 和 Embedding 两版。

**完整操作文档：[新集群交接文档](HANDOFF.zh-CN.md)**。从公开数据与干净基座开始，包含环境、数据筛选、3D 标签生成、保存/恢复 smoke、70k 训练、4/2 GPU 正式评测和原始结果验收。

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

布局：

- `runtimes/baseline/`：独立历史基线源码，不是新版关闭 loss。
- `runtimes/query_fixed/`：独立 FIXED 源码与固定教师。
- `runtimes/trajectory_v2/`：最新 Query/Embedding 3D v2，共享实现、分开配置。
- `configs/`：完整科学配置模板；`scripts/`：可迁移入口。
- `assets/`：冻结任务清单、训练 registry、归一化统计；没有训练数据和模型参数。
- `provenance/`：来源与发布文件 SHA；`tests/`：CPU 迁移边界检查。

现有训练 checkpoint、完整 optimizer/RNG 状态以及已生成的约 350.5 GB 3D 标签不在 GitHub 中。可从旧集群搬运自有产物，或按文档从公开输入重新生成/训练。不要把只含 safetensors 的模型导出当作完整续训。

本仓库聚合了 StarVLA、OpenPI 和少量 RoboCasa/评测适配代码；各自许可证及署名保留。见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
