# B1500产出后的有界fslgpu评测（r9）

r9在fslgpu等待r8的完整1500步训练完成凭据，最长4小时。等待期间仅CPU读取共享
状态，不加载模型、不占用GPU推理、不创建/恢复任何SCO任务，不替代原生十分钟续训。
若B训练HOLD或等待超时则保存原因并停止，不循环重试。

凭据必须匹配B工程run、1500步、原训练release SHA，并存在完整保存marker。
fslgpu有其他计算进程时等待其自然结束，不停止他人进程。就绪后持有共享训练锁，
核验实际单H100/CUDA-EGL UUID映射，调用已经实际验收的r7评测入口，严格加载
B1500完整权重，完成两个原协议回合，保存原始success/trace、实际三视图和GDB证据。

成功后写本目录control/FSLGPU_EVALUATION_ACCEPTED.json，包含A1500既有证据与
B1500新证据，both_short_steps=1500、full_benchmark=false、ACP_bug_fixed=false。
只有此文件的当前实例身份和原始证据核实后，才能说两版1500步有界验证全部完成。

本地凭据拒绝测试、最终部署SHA及远端真实CLI dry-run已通过；B1500尚未产出，
因此没有伪造该权重GPU验收。17:07在当前fslgpu实例启动，PID542；17:11处于
waiting_for_complete_B1500。如果CCI实例被暂停，进程不会在别的宿主自动冒充继续；
专用只读检查会按实例hostname/start识别，并交回主对话处理。
