# Pri4R首段物理时间缺口修复

2026-11-01。此项只调整3D辅助监督的时间有效性；保留RoboCasa365原始RGB、状态、action、episode/frame编号与抽样分布。

## 原因

上游RoboCasa采集代码先保存reset状态，再通过`env.env.step(zero_action)`执行一次不记录的dummy step，最后记录第一个真实动作之后的状态。已有4条真实示范的时间序列都是首段2dt、其余dt。例如20Hz时：

| 状态 | simulator物理时间 | 导出dataset时间 |
|---|---:|---:|
| reset快照/记录row0 | 0.50s | 0.00s |
| dummy之后/真实action0之前 | 0.55s，未记录 | 无 |
| 记录row1 | 0.60s | 0.05s |
| 记录row2 | 0.65s | 0.10s |

旧GT只将等间隔dataset时间传给训练reader，尽管生成阶段已经记录物理间隔异常。因此row0的0.10秒差分被当作0.05秒目标，后续目标相对row0也整体晚一个时间槽。其根源是源采集记录，而旧reader未屏蔽这段不完整物理horizon。

## 修复行为

- 新GT保留`timestamps.npy`用于校验原dataset行，另存真实`states[:,0]`为`simulator_timestamps.npy`并校验SHA。
- 新策略`physical_window_alignment_v1`同时检查每个端点相对当前帧的物理时刻和相邻间隔。
- 只有确实含首段2dt的`start=0`辅助窗口整段mask；正常首段dt不会被屏蔽。从row1开始，物理时间偏移为常量，其正常窗口保留。
- 不把第一段除以2、不插值补状态、不重新模拟缺失状态、不移动标签槽、不删除整个action训练样本。
- `point_current`仍保留真实当前坐标；被屏蔽的`point_delta`零填充、`point_valid=false`，这些零不是监督目标。
- 全局无有效点目标时辅助loss和其梯度为零，action训练及optimizer/scheduler/EMA仍正常推进。已有Adam动量仍可能移动辅助参数，这不等于发生了辅助梯度泄漏。

未知的内部缺帧、时间倒退及其它首段间隔继续拒绝，不自动将它们当成已知dummy缺口。

## 数据和状态兼容

GT生成器升级为`rigid_surface_replay_physical_clock_v2`。GT配置、生成记录、数据reader、模型输入metadata、point-head语义signature和完整checkpoint recipe均绑定新策略。

**旧GT完成凭证、旧resolved配置和旧训练state不能直接接着使用。** 保留原目录，按[训练手册](PRI4R_PAPER_TRAINING.zh-CN.md)在新资源/GT输出目录执行prepare→generate→verify，再生成新的resolved配置。模型/base、norm及原始示范数据可以复用原封存资产；不要覆盖旧GT或手改旧checkpoint marker。已有正式训练若需要迁移，必须另做明确的状态兼容处理；本次没有迁移或重启任何正式任务。

修复没有重写官方action数据的历史首行采集方式。它消除的是缺失中间状态造成的错误3D辅助时间监督。

## 验收

最终结果见[时钟修复验证记录](../provenance/PRI4R_PHYSICAL_CLOCK_VALIDATION.json)。包含正常首段不屏蔽、2dt首段全horizon屏蔽、EOF/缺失/损坏/内部gap拒绝、模型及checkpoint旧策略拒绝、全无效和部分rank无效的真实4进程CPU梯度归约与审计。

真实数据核对范围固定为原有4条示范1126帧；验证各原始坐标/采点数组SHA不变，每个正常窗口的current/delta/mask逐值一致。单H100短训/保存/新进程恢复/推理使用新封存源和新GT。这里不包含全量GT、正式30k训练或benchmark结果。

实际结果：1126个窗口中仅4个异常start0被屏蔽，其余1122个窗口逐值不变；原七类采点/坐标数组SHA不变。H100第1次更新的action loss与旧版相同，有效XYZ恰好少153600（一个1024×50窗口）。3次更新/完整保存、新进程2→3逐值恢复、EMA、无3D推理和实际并发服务均通过。

修复后再次逐项核对论文公开结构、监督、训练配方和推理路径，没有发现其它具有明确作者证据的必修偏差。作者未公开且已由用户选定的空间/容量/归约等约定保持固定；RoboCasa365迁移不等于论文原Human-50分数复现。
