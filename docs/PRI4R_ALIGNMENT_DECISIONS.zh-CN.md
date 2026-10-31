# Pri4R 对齐候选：已确定项与待选择项

2026-10-31。目标是在RoboCasa365复现Pri4R带来的性能增益。论文明确的要求优先；论文未说明的π0.5本体行为，以固定的[官方OpenPI PyTorch commit 215abfb](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)为参考。作者尚未提供训练实现，不能保证复现其未披露细节或最终增益。

用户已确认1A/2C/3A和世界坐标。最终采用XYZ mean L1、已有4条示范覆盖诊断选定边长3m、保留2层辅助头。本组合已重新通过单H100训练/恢复/推理、官方对照和服务端验证。当前运行说明与证据以[训练评测手册](PRI4R_PAPER_TRAINING.zh-CN.md)为准；下方保留决策过程。

2026-11-01更新：已修复源首段2dt导致的辅助horizon时间错位，新增物理时钟契约和首帧整窗mask；新组合27项CPU、1126真实窗口及H100训练/恢复/推理/服务验收通过。旧GT完成凭证和旧时序策略state不直接复用，按[时钟修复说明](PRI4R_PHYSICAL_CLOCK_FIX.zh-CN.md)生成新GT和resolved配置。下方10月31日的验证数字为历史过程。

## 已确定并实施的部分

- 用户已选择世界坐标、单位米；当前点和整段未来位移统一使用world frame，不额外标准化。已有GT坐标本身是world tracks；新的物理时钟契约另要求在新GT根生成并核验，不复用旧完成凭证。坐标及时间策略进入head语义签名，禁止旧base-frame或旧时间策略训练state混入。
- π0.5主干保留官方selective BF16/FP32，原生DDP、无外层autocast，不经过DeepSpeed整体BF16转换。新增辅助头FP32是明确的本地实现约定。
- 使用原生torch AdamW统一参数组，beta=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、clip=1；没有ZeRO FP32 master副本，Adam状态遵循实际参数dtype。
- 原始joint attention不因辅助头开启而拆分。训练读取prefix输出，不改原action mask，不把辅助query插入动作prefix。
- 完整保存/恢复支持DDP原生model、Adam、每rank RNG、scheduler/data进度、EMA；实际恢复逐值审计，拒绝ZeRO/DDP混合状态。旧的FP32-master审计不冒充新路径的证据。
- 论文要求EMA=.999保留，以每次optimizer update后的全部唯一模型参数更新FP32 shadow，不作warmup或bias correction；同时保留raw和EMA。官方PyTorch没有EMA，不能称该扩展是作者实现。
- 严格加载完整权重后，推理物理删除辅助模块，再把原动作模型移至GPU；无新增3D输入。
- 论文给出的global128、30k、peak LR5e-5、warmup10k、cosine、1024点和loss权重1保持。未披露的LR末值采用官方通用CosineDecaySchedule默认2.5e-6，decay_steps=30k。它是选定的官方默认，不是作者确认值，也不混用LIBERO的专用配置。

## 已确认的选项（保留备选理由）

| 项目 | A | B | C | 推荐与证据边界 |
|---|---|---|---|---|
| L1归约 | 对有效XYZ分量平均 | XYZ求和后，对有效点/时间平均 | — | 推荐A，与常见PyTorch L1默认一致；B在同mask下精确放大3倍。作者未披露。 |
| cube范围 | 边长4m（当前半边长2m） | 边长2m | 先用已有4条示范检查2/3/4m覆盖和位移分布，再定一个统一范围 | 推荐C，不生成全量GT、不使用未来运动筛点或robot/object配额。 |
| 辅助容量 | 2层、d2048、8头、FFN4096；Point3→128→2048；Fusion4096→256→3 | 同宽1层与2层做短训比较后选择 | — | 推荐A；图2支持self→cross及重复L，但没有披露L等规格。不能证明减层会更好。 |

已按用户1A/2C/3A执行有界验证并将3m写入最终配置；没有全量GT或正式训练。

## 可直接固定的其它约定

H50沿用π0.5的动作horizon，51个位置形成50段相邻位移；缺失未来点监督mask掉，不凭空标成零位移。原动作loss、动作EOF、分词/归一化/相机处理维持已核对的π0.5/RoboCasa365适配。取final norm后的有效prefix，包括π0.5原有的proprio文本token。未引入累计位置loss、RoPE、causal temporal attention、teacher、Memory或辅助权重自适应。

RoboCasa365的300任务/100demo仍属于用户指定的数据集迁移；论文是24任务/Human-50。后续判断“Pri4R是否带来增益”，baseline与Pri4R应使用同一base、数据、官方精度、优化/训练预算及评测协议，只开关辅助分支。本阶段不开展大规模训练。

## 当前验证边界

16项CPU测试通过：6项新增原生混dtype/世界坐标/launcher/discard/epoch边界检查、6项配方/GT范围契约、3项辅助数学/EMA检查、1项四rank CPU Gloo归约；另7项legacy契约通过。原生状态测试实际调用Accelerate 1.5.2保存和恢复FP32/BF16小模型与Adam，并验证删除Adam状态会被拒绝；不是完整π0.5或GPU测试。

修正了完整积累窗口预取恰好到epoch末尾时Accelerate提前同步的边界：固定sync_with_dataloader=False，测试保证只在完整积累窗口末端更新。训练不强制开启官方未启用的deterministic_algorithms；固定seed与完全确定性算子不是同一件事。

fslgpu使用原SCO bootstrap后实际H100 CUDA初始化通过；直接裸SSH环境出现的CUDA803已通过正确库环境消除。最终组合已通过完整模型GPU训练/恢复/推理、18层辅助独立梯度、官方动作前向和805梯度逐值对照、B32 CUDA EMA容量及实际服务端测试。四个CPU进程的原生混dtype状态恢复通过；4张实体GPU尚未联跑。
