# 参考聊天处理核对：2026-10-23

参考聊天：`ours未来Query版PI0.5`，ID `01a0d4a3-cb83-78a3-933b-5de7a4a78a6e`。
已读取实际聊天记录、修复提交 `040f08a10088d810a265a7f9afea568d4b43a9a3`、
双5090任务 `pt-qykfd96c` 的独立GPU验收文件，并与当前r4及真实失败worker证据逐项对照。

| 参考修复 | 当前r4代码和失败现场 |
|---|---|
| 每task绑定对应的EGL设备 | 已实现；EGL0 UUID等于已分配CUDA0 |
| 模拟器清除继承的CUDA_VISIBLE_DEVICES | 已实现；实际值为空 |
| OMP/MKL/OpenBLAS限制为4/4/1 | 已实现；实际值一致 |
| CUDA/EGL命名空间分别校验 | 默认映射、显式UUID映射相同；3类错误映射均拒绝 |

执行 `reference_thread_audit.py` 的CPU/源码对照已通过。参考GPU验收是baseline、
双5090、4个完整回合；不能替代当前A1500在双H100上的EGL验收。
当前真实SIGABRT仍是 `mjr_readPixels -> libnvidia-eglcore.so.580.95.05 -> abort`。
因此当前不是漏接该聊天的设备绑定/线程限制修复，不能重复应用它们后宣称修复。

该聊天后来发布的固定场景、多环境、CUDA Graph流水线解决可比性和吞吐；其当前
公开variant注册表支持baseline、query_fixed及两种trajectory_v2，不含这次v3 A/B。
不能直接替换目录或套用H100 n_envs=12/24的利用率结果。本次没有修改模型、权重、
损失、训练或评测协议，也没有新提交GPU任务。

采用其验证方式：下一项诊断以已验收的baseline入口为对照，在同一worker、同一
模拟器环境分别运行baseline和A1500，再区分通用渲染问题与A轨迹触发的差异。
若接入已验收的新评测入口，只为新的v3推理实现做一次有针对性的短测；以后同实现
换权重复用验收，不重复完整多环境smoke。当前已有原生崩溃仍需真实回放验证修复，
不能以“只换权重免smoke”绕过故障。固定场景协议如被接入，需与旧串行结果分开记录。

完整源码SHA、执行对照、参考GPU证据身份和当前失败worker事实见
`REFERENCE_THREAD_AUDIT.json`。当前A1500保留、B1500未开始、训练HOLD保持。
