# EXP-005：PI0.5 + Pri4R论文方法训练准备

本实验按Pri4R论文公开方法独立实现，并迁移到RoboCasa365。它与EXP-003/004的Query、arm/object配额、累计或位置监督对照独立；不把那些版本的成绩算作本实验结果。

- 运行手册：[资源、少量GT、单卡验收、四卡训练和续训](../../docs/PRI4R_PAPER_TRAINING.zh-CN.md)。
- 模型代码：`runtimes/pri4r_paper/`；模型配置：`configs/pri4r_paper.yaml`。
- GT配置：`configs/pri4r_paper_gt.json`；生成入口：`scripts/prepare_pri4r_gt.py`。
- 发布验证记录：[PRI4R_PAPER_VALIDATION.json](../../provenance/PRI4R_PAPER_VALIDATION.json)。

本次授权范围仅为训练准备及短测：4条真实示范、1126帧、每条1024点，3次优化更新和新进程恢复。没有全量GT生成、正式训练或新的benchmark成绩。作者未公开的细节列为实现约定，不宣称与未公开作者代码完全一致。

单H100短测使用B4×acc32=global128，AdamW及FP32 EMA放CPU；四卡正式配置使用B32×4×acc1=128，状态放GPU。四进程CPU归约检查与真实四卡NCCL/容量检查是不同的证据，验证记录分别标明。
