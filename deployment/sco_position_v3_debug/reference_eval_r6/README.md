# r6：参考入口与评测分配器对照

2026-10-23用户要求直接对照已验收聊天代码，并检查是否缺少图形能力。
SCO实时API确认失败任务pt-xx84awzx、原生抓栈pt-max2mytx及参考通过任务
pt-qykfd96c均为同一镜像，且提交环境包含NVIDIA_DRIVER_CAPABILITIES=all。
原失败进程实际已加载NVIDIA EGL580.95.05并生成三路有效RGB，因此缺少该提交
变量不符合现有证据；此结论不等于完整证明宿主驱动与所有库都兼容。

远端实际源码核对：baseline与A的simulation_env.py逐字节相同；参考入口通过
显式PYTHONPATH加载的RoboCasa/Robosuite，正是A runtime链接到的同一份sim_code。
gpu_identity.py完全相同；审计包装差异仅新增环境字段和每100步进度落盘。
两边均明确逐卡EGL绑定并使用OMP/MKL/OpenBLAS=4/4/1。

发现的可测试启动差异：当前A评测继承训练的
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，而参考聊天后续已验收的
fixed2500评测bootstrap明确移除该变量。r6将其作为候选原因，不预先断言因果。

同一worker、同一CUDA0/EGL0依次运行：

1. A1500，保留当前分配器设置，r4入口。
2. A1500，仅清除PYTORCH_CUDA_ALLOC_CONF及其别名PYTORCH_ALLOC_CONF，仍用r4入口。
3. baseline70k，使用已验收040f08a1的原入口及默认分配器作参考。

每项CloseBlenderLid、seed7、2完整回合、900步horizon、每5步重规划。保留原生
MuJoCo/EGL和相同精度，不缩短回合或省略渲染。每次独立进程、严格完整权重加载、
原始success/trace与GDB原生栈并存。清除分配器只作用评测子进程，不改训练recipe、
优化器、模型或已有权重。baseline只用于诊断，不与A1500比较方法性能。

主控制器保留原训练HOLD，持有只读训练锁及独立诊断锁。最多三次回放；如果worker
被外部终止，native10可重新启动控制器，但控制器会记录不混合worker的HOLD并停止，
不把不同worker结果拼成受控对照。三个case完成也只写reference_comparison_complete，
bug_fixed保持false，须主对话解释实际证据；不会自动恢复训练或启动B1500/70k。

本地3项环境隔离测试通过。最终10文件SHA部署已核验；fslcpu上3组真实子进程环境、
三个CLI入口及GDB正常退出/受控abort/超时清理全部通过。第一次CPU验证因CCI实例
被暂停中断，恢复后已完整重跑通过，不把中断当作代码故障。实际ACP worker仍会
重新执行CPU检查，再核验NVIDIA_DRIVER_CAPABILITIES和两H100的CUDA/EGL UUID。

提交：标准实际MEMBER_QUOTA_EXCEEDED（剩余GPU0）后保持2H100规格转SPOT。
job：pt-7esakt2h；native：f5083efb-0e8c-4e72-a7df-c802d9602cf6；autoRetrain=true/10min。
13:54 API worker Pending；尚无新GPU对照结果，不宣称修复通过。

manifest SHA256：cdb0870b51b6acf60ecab847805ff1f3669fd9bfe11de07af936b63d7d9d2a5f。
原始API与CPU验证分别见CAPABILITIES_AUDIT.json、CPU_PREFLIGHT_READBACK.json；
提交凭据见spot_submit.json。原r2/r4/r5的部署文件、状态与训练HOLD均保留。
