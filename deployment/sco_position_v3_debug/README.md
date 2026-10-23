# Original SCO bounded A/B validation

These launch files are specific to the original SCO installation. Portable
training and evaluation remain in `scripts/`; do not run this controller on
another cluster without translating its explicit paths and preparing its
release manifest and resolved configurations.

Revision r1, 2026-10-21, preserves A/B model, objective, data, optimizer and
the 1500-step engineering limit. It adds the system NVIDIA driver library
directories to the launch environment. The old launch selected the image's
CUDA 12.6 compatibility library and failed with CUDA Error 803 on the real
ACP worker before the first training update.

On the CPU access container, separate-process `dladdr` checks resolve the old
path to `/usr/local/cuda-12.6/compat/libcuda.so.1` (driver API 12060) and the
corrected path to `/lib/x86_64-linux-gnu/libcuda.so.1` (API 13000). This checks
library selection, not GPU initialization. The controller records both
environments in independent GPU probe processes and only continues after the
corrected environment initializes two H100 devices and performs an allocation.

An application error writes `HOLD_FOR_MAIN.json` and exits without a blind
retry loop. Platform success is therefore not application acceptance.
The old task's full error record is preserved. Only after confirming its
worker had exited, the pipeline lock was free, and no training recipe or
checkpoint existed was r1 deployed and submitted.

Remote evidence and release manifests are stored in the experiment root's
`control/` and `repair_r1/`. The immutable r0 archive and its manifest are
retained separately. The r1 manifest covers 964 deployed files.

Revision r2, 2026-10-22: the actual r1 worker verified the CUDA correction with
separate old/fixed subprocesses and passed a two-H100 NCCL all-reduce. It then
failed before the first training update because the real RoboCasa overlay
preceded the policy import shim during lazy model construction. The policy
environment consequently attempted to import MuJoCo. V3 policy paths now put
the shim first; simulator calls still exclude it and import the real simulator.
Historical variants retain their previous import order.

The exact lazy OpenPI configuration/model imports pass on SCO for both A and B;
the simulator also imports under evaluate.py's actual subprocess environment.
Local validation passes 21 repository tests and two collector regression tests.
The corrected smoke and short run names end in `_r2`; the old zero-update
recipe remains preserved. No learned state was discarded. The r2 remote
manifest covers 965 files. GPU training/save-restore, action inference and EGL
rendering remain pending. The collector can audit a terminal last-known job
after native monitor removal without claiming it has a live recovery owner.


#### 2026-10-23 12:37：A/B真实原生栈已捕获，故障进入NVIDIA EGL内部

诊断任务pt-max2mytx于02:12:35在host10.115.72.99开始，02:17:29第一次真实A1500回放捕获SIGABRT后立即结束；12:37主对话已核对API job/workers、严格workload日志、attempt 20261022T181241Z_8003ae9c的hostname/start、TERMINAL和原生栈。此次不是CPU预检中的人为abort，也不是超时中断。平台SUCCEEDED表示诊断保护退出，native已移除，不能称评测通过。

真实主线程栈为mjr_readPixels -> libnvidia-eglcore.so.580.95.05多层内部调用 -> abort -> raise。EGL0 UUID 4ccc10e5-a575-a988-a01d-777fd125ac48与已分配CUDA0一致，未见选到未分配卡的证据；三路256x256RGB有效，最后落盘200模拟步骤/40RPC，并非精确崩溃步。此前完整通过的fslgpu回放同样报告NVIDIA580.95.05，因此可确定原生崩溃位置，但不能仅凭版本判定驱动缺陷，驱动内部问题、上下文使用和库交互仍待区分。完整10文件已经回传并逐文件SHA验证；未完成回合，未生成可计分的成功率。

读取阶段fslgpu已SUSPENDED，启动实际遭成员GPU配额0拒绝；恢复现有fslcpu后SSH可用，仅CPU读取RobotVla内证据。发现本地诊断collector在SSH失败时保留旧latest排队快照，已修复为发布当次API状态并明确应用证据不可用，4项身份/历史作用域/传输失败测试通过。仅更新本地collector及记录，未修改已部署r2/r4/r5运行payload或再次提交GPU任务。

A1500权重继续保留；B smoke3通过但B1500目录仍不存在。原训练HOLD保持，训练与诊断writer锁都已核实空闲。下一步需围绕已捕获的EGL原生栈做受控复现和修复验收，当前不宣称错误修复或算法效果提升。证据：work/pi05_embedding_position_v3_20261020/checks/20261023_123719及native_debug_r5/evidence/acp_20261022T181241Z_8003ae9c。
