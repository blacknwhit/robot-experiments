# EXP-008：FOCAL-VLA 迁移到 RoboCasa365

目标是用子任务相关的几何蒸馏、未来运动蒸馏和语义辅助监督训练 π0.5。完整运行说明：[8 卡全量训练](../../docs/FOCALVLA_8GPU_TRAINING.zh-CN.md)。用户确认的本轮配置为 **8 卡、global64、70k，每10k保存**；本次交付准备训练代码，未启动正式70k。

## 方法与配方

依据用户提供的 [FOCAL-VLA 论文](https://arxiv.org/abs/2609.21228) §III、§IV-A。代码保留当前观测输入、8+8 query、冻结 VGGT/Track4World 的完整帧教师输入与实体区域池化、原生语义 CE、动作对 query 的条件读取。推理无未来帧和GT子任务输入。

标注采用此前验证的人工粗模板＋离线物理事件增强，解决小物体和遮挡的识别问题；这改变了论文的默认 Qwen3-VL-8B 标注工具，监督信息类型和模型训练路径不变。此处是论文方法的 RoboCasa365 迁移，不把 70k/global64、增强标注或采样细节写成作者原配方。

入口：[科学配置](../../configs/focalvla_robocasa365_8gpu_70k.json)、[资源模板](../../configs/focalvla_resources.example.json)、[torchrun 启动脚本](../../scripts/focalvla/train_8gpu.sh)、[完整实现](../../runtimes/focalvla/focalvla/)。

## 已有事实

| 范围 | 结果 |
|---|---|
| 离线标注覆盖检查 | 已有300类各一条的审计；不等同全30,000条均已人工审核 |
| 前一版本两卡短训 | global128，micro16/acc4，完成50步并从25步恢复；中位4.308秒/步，旧版17.271秒/步，约4.01倍加速 |
| 两卡运行质量 | 全参数主干/辅助头梯度与同步检查通过；checkpoint包含FP32 Adam状态、两rank RNG及6400抽样进度；固定工程验证损失0.969393→0.928861 |
| 方法数值核对 | 共享context数学参考821项梯度相对L2 2.16e-6；BF16优化梯度cosine 0.998607；改变GT semantic标签不改变动作/几何/运动输出 |
| 本次全量计划 | 300任务、30,000条；4,480,000 draws，4,076,006 unique episode/frame；含4条不足50帧演示，末端补齐 |
| 正式结果 | 全量教师缓存、实体8卡速度、70k checkpoint和策略成功率均待实际执行 |

上述速度和损失来自工程子集，不能据此宣称论文成功率已复现。最终比较需要正式权重和统一 RoboCasa365 评测协议。

## 2026-11-15：portable 整理

复用已验证的前向与 FlashAttention 核心，去除服务器绝对路径及两卡/50步/旧checkpoint绑定。新增全量确定性采样、episode缓存、有界读取、可配置DDP、完整断点恢复。资源JSON只绑定路径；源码与实验输出分开。

本次 portable 已在单 H100 上用真实演示依次跑通：采样计划 → 离线标注 → 两个冻结教师 → 缓存封存 → DDP 训练第1步 → 完整保存 → 新进程冷恢复 → 第2步 → 完整保存。测试使用1条演示、4个缓存时刻、global2；四项损失和梯度范数有限，峰值已分配显存57.90 GiB。训练用的12张缓存RGB与源视频逐像素一致，轨迹末端重复最后一帧的测试通过。CPU的8-rank数据分片、70k学习率端点、不完整/不兼容checkpoint处理、精确AdamW/RNG续训4项测试通过。完整取样计划确实覆盖全部30,000条演示。

这次测试验证了整理后的执行链路与冷恢复，不能替代实体8卡吞吐或70k效果验收。详情见 [机器可读验收记录](validation_20261115.json)。

## 来源

OpenPI兼容实现及必要三视角修改随代码保留，许可证见 [OPENPI_LICENSE](../../runtimes/focalvla/vendor/OPENPI_LICENSE)。教师分别来自 [VGGT](https://github.com/facebookresearch/vggt)、[Track4World](https://github.com/TencentARC/Track4World)；固定commit及必要的DA3导入处理见 [provision_teachers.py](../../scripts/focalvla/provision_teachers.py)。权重、数据、模拟器资产和机器资源路径不上传本仓库。
