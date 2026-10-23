# EXP-001：原PI0.5基线

[返回实验总览](../../EXPERIMENTS.md)

## 实验定义

在RoboCasa训练原PI0.5动作策略，作为后续辅助任务方法的基线。配置`baseline`，源码`runtimes/baseline/`，模板[baseline.yaml](../../configs/baseline.yaml)。无Query、无未来视觉或3D辅助分支；历史GC=false，不借用后续配置的GC开关。

初始化为原始OpenPI `pi05_base`转换后的`model.safetensors`；训练至70k。完整配方以本配置和历史封存配置为准，不应用EXP-004的模型、GT或训练入口约束。

## 已有结果

历史70k完整测试：**391/2500，15.64%**，50 tasks × 50 trials。该结果来自旧集群，不是本仓库在新集群重跑的分数。权重和完整optimizer/RNG状态不在GitHub。

## 资源与执行

- 复用已有数据、原始预训练权重、norm、tokenizer和匹配环境；本实验不需要3D GT。
- [历史部署/训练/恢复手册](../../HANDOFF.zh-CN.md)中的`baseline`配置是本实验入口；资源已就绪时跳过安装和下载。
- [原始迁移来源](../../provenance/PORTABILITY.md)、[公开包初期工程验证](../../provenance/VALIDATION.json)是代码来源/工程证据，不替代完整benchmark证据。
- 新固定场景评测用[公共评测协议](../../docs/ASTRA_FIXED_SCENES.zh-CN.md)，指定本实验权重和独立输出目录；旧391分单列。EXP-004的输出不得放进本实验结果中。

本次仓库按实验归档只补文档入口，没有移动、恢复或重跑历史任务。新增复现运行需记录独立run-id、源码/配置/场景SHA及结果出处。
