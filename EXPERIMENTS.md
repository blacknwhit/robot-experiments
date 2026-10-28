# 实验总览

本文件是全仓库实验入口。**按一次研究实验管理；一次实验可以包含多组对照。** 下面的EXP编号是本仓库的管理编号，历史训练名称、原始记录编号和证据身份继续保留。

| 编号 | 实验问题 / 内容 | 内部配置 | 状态与实际结果 | 独立文档 |
|---|---|---|---|---|
| EXP-001 | 原PI0.5 RoboCasa基线 | `baseline` | 历史70k：391/2500，15.64% | [PI0.5基线](experiments/exp001_pi05_baseline/README.zh-CN.md) |
| EXP-002 | Query-FIXED未来视觉辅助监督 | `query_fixed` | 历史70k：559/2500，22.36%；当前这里记录的最佳FIXED | [Query-FIXED](experiments/exp002_query_fixed/README.zh-CN.md) |
| EXP-003 | 3D轨迹v2：比较Query与Embedding | `query3d_v2`、`embedding3d_v2` | 历史70k：Query 373/2500（14.92%），Embedding 443/2500（17.72%） | [轨迹v2](experiments/exp003_trajectory_v2/README.zh-CN.md) |
| **EXP-004** | **本轮3D位置轨迹：监督与结构对照** | **A：`embedding3d_pos_a`；B：`embedding3d_temporal_b`；C：`query3d_temporal_c`** | **A/B已有有界GPU工程证据；C仅CPU结构/接线检查。三组尚无正式70k成绩，不宣称效果提升** | **[本次实验：资源复用→GT→三组训练→测试](experiments/exp004_trajectory_v3_abc/README.zh-CN.md)** |

历史分数来自旧集群原协议，不是本公开代码在新集群重跑的分数；新的固定场景集合不能直接冒充旧逐场景对照。新评测须记录实际checkpoint/config/source/scene身份和每任务原始结果。

## 管理约定

1. 每次新研究实验新增`experiments/expNNN_name/`，维护自己的README、方法说明、运行状态、配置映射与结果出处，再在本表新增一行。本轮A/B/C是EXP-004的三组，不拆成三个无关联实验，也不改写EXP-001/002的结论。
2. 数据、**原始OpenPI `pi05_base`预训练权重**、norm、tokenizer、已验证环境、兼容GT和统一任务/episode种子定义按身份复用。当前评测无需PLANS。新实验不意味着重新下载/安装；“从base开始”是预训练模型微调，不是全模型随机初始化，更不是拿历史70k权重续训。
3. 各实验配置、日志、checkpoint、评测输出和发布记录独立。EXP-004使用`--home H --experiment-root E`：H是已有公共资源，E是本次实验产物根；E内部各对照组再使用独立run-id。旧实验产物保留原路径和原release，不通过文档整理搬动。
4. 公共实现仍放在`runtimes/`、`scripts/`，科学配置模板放在`configs/`，各实验文档显式映射所用版本。逻辑分离不要求复制多套相同脚本，更不能隐式改变旧模型。
5. 每次实际运行保存code commit、source manifest、解析后的完整配置、环境freeze、数据/GT/场景身份和任务/原生monitor标识。新代码不冒充旧运行release；恢复继续使用原封存组合。
6. 结果明确区分CPU检查、GPU工程验收、短训和正式benchmark。实验记录模板见[模板](experiments/TEMPLATE.zh-CN.md)。

## 公共资料

- [历史PI0.5/FIXED/v2部署参考](HANDOFF.zh-CN.md)：资源来源、旧配方和空白集群准备；不会替代各实验的独立手册。
- [固定2500回合高吞吐评测](docs/ASTRA_FIXED_SCENES.zh-CN.md)：每卡一份常驻模型，多task共用并行环境；具体checkpoint与输出目录由各实验指定。
- [多版本模型接入与选择](docs/EVALUATION_MODELS.zh-CN.md)：已登记版本、配置/权重映射、模型侧适配及各版本验证边界。
- [第三方许可证](THIRD_PARTY_NOTICES.md)、[来源记录](provenance/PORTABILITY.md)。GitHub不保存大数据、GT数组或模型权重。

本总览的评测操作说明更新于2026-10-28；实验历史结果仍保留各自日期。这不是SCO实时任务看板。
