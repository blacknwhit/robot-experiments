# fslgpu A/B实际评测验收（r7）

用户2026-10-23要求直接在fslgpu测试，尽快准备两版训练和评测。当前实例为
app-3d35655f960040e19546745dbd3e2cd4-68997f79fd-gsmzn，单H100，实际CUDA/EGL UUID
275aa92c-a9cc-4920-0cd6-1711f28cd003。r7只评测，不在登录实例执行训练。

实际验收完成：A1500原分配器、A1500默认分配器、B smoke3默认分配器各2完整回合，
合计5400模拟步骤/1080策略请求；全部GDB正常退出，无原生信号或超时，三视图有效。
A两种设置的两条完整episode trace完全相同。A严格加载861keys，B严格加载885keys；
每项均匹配checkpoint SHA，无丢弃权重。41个原始证据文件已回传并逐文件SHA核验。
详细结果在INDEPENDENT_FSLGPU_ACCEPTED.json与evidence/20261023T083305Z_2da6ee86。

每项原始success均为[false,false]。这是短训练工程验收，不是正式benchmark，不据此
评估方法提升。两种分配器都通过，不能认定分配器是先前ACP原生崩溃的根因，也没有
证明ACP渲染稳定性已经修复。

此前r6 ACP原job pt-7esakt2h在模型服务启动阶段被中断，worker Deleted，尚无模拟器
输出/完成结果。native生成的pt-msv0o6e2仍Pending。按本次转向fslgpu的指令，已先停止
精确native f5083efb-0e8c-4e72-a7df-c802d9602cf6，再停止其currentJob pt-msv0o6e2，
复核worker Deleted；原始记录全部保留，不重复启动诊断。

本次实际证明：A1500训练后评测可用；B的两rank训练/完整恢复smoke此前已通过，
本次进一步通过B完整权重的真实环境评测。B1500短训及其训练后评测尚待完成。
原r2训练模型、GT、loss和配方未改。r4评测入口复用，默认分配器只影响评测子进程。
