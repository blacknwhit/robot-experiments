# EXP-002：Query-FIXED未来视觉辅助监督

[返回实验总览](../../EXPERIMENTS.md)

## 实验定义

比较在PI0.5中插入Query并加入固定教师未来视觉表征监督的效果。配置`query_fixed`，源码`runtimes/query_fixed/`，模板[query_fixed.yaml](../../configs/query_fixed.yaml)。历史实现使用64个Query、G8采样和固定初始教师；本实验不是50 Query的3D点预测方法。

初始化为原始OpenPI `pi05_base`，本实验训练到70k。教师/完整优化器状态属于本实验保存与恢复契约；不套用EXP-004的无教师配置。

## 已有结果

历史70k完整测试：**559/2500，22.36%**，为当前这里归档的最佳Query-FIXED结果。最佳点为70k，不是100k。该分数来自旧集群原场景协议，不能混入新固定场景结果。

## 资源与执行

- 数据、原始预训练权重和匹配环境可与EXP-001复用；不需要EXP-003/004的3D GT。
- [历史部署/训练/恢复手册](../../HANDOFF.zh-CN.md)中的`query_fixed`入口属于本实验，已有环境无需重装。
- 来源和重构边界见[迁移记录](../../provenance/PORTABILITY.md)；历史主仓commit缺失等边界保留，不把公开包CPU检查称为历史逐位复现。
- 使用[固定场景评测](../../docs/ASTRA_FIXED_SCENES.zh-CN.md)时选择本实验70k权重和独立输出，与其它方法同场景重测后再作配对比较。

本次管理调整未修改历史训练配方、迁移权重或重跑任务。后续运行记录独立run-id、教师/optimizer/RNG恢复证据及配置/场景身份。
