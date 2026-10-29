# 全黑观测：先定位出错层，不直接归因显存或多卡

当前`All-zero RGB camera frame`表示传给观测检查的数组全部为零，**不等于已经确认GPU渲染失败**。公开pipeline会在一路异常后取消其它路，其余`cancelled`不是独立黑图故障。

## 远端Astra只需读取现有失败产物

更新代码后，用已有Python/NumPy环境运行以下命令，OUT填写失败任务实际输出目录（包含`parallel-runs`和`EVALUATION_IDENTITY.json`的目录）：

```bash
python diagnostics/inspect_black_frame.py "$OUT" \
  --task WaffleReheat --episode 0 > waffle-black-frame-report.json
```

把生成的JSON交回主对话。该工具只读现有文件，不加载模型、启动GPU、修改seed、修改校验或重跑全量评测。不存在对应文件时输出`no_matching_public_pipeline_artifact`，不将缺证据当通过。它读取失败帧的三路数组、phase、初始观测marker、运行身份，以及每批实际记录的CUDA/EGL编号、任务分组和模拟器命令；编号相同不等于物理UUID已核对。正式评测不保存initial.npz也正常，可读取`env0.initial.json`的health记录。

## 单卡正常、多卡失败时

先对失败多卡OUT和正常单卡OUT分别运行上面的只读命令。单卡没有对应错误文件时仍会输出运行身份和批次分配，不把没有错误当作该场景已经完成。确认两边确实包含相同task/episode、相同模型与seed，并核对n_envs、RGB选项和代码身份。

当前公开入口每卡独立策略服务、独立RenderOwner、独立CPU worker池和Pipe；各slot使用新建的环境变量dict，没有发现多线程改写全局GPU环境变量的路径。8卡n_envs16意味着最多128个CPU物理worker。任意一路故障会取消其它路，不能把后续取消计为独立黑图。

这条对照提高了以下检查的优先级，但尚不能确定根因：

1. CUDA与EGL设备表是否映射到相同物理UUID/PCI地址。CLI只校验数量和唯一性；GPU0单独正常不能证明GPU1–7的映射正确。
2. 全机CPU、内存、共享内存和驱动资源压力，以及多个渲染进程同时运行的行为；没有OOM证据时不能直接归因显存不足。
3. 多卡的任务分组和各worker前序场景不同，可能改变上下文创建/释放历史；固定seed不排除实现受到进程历史影响。

后续最小GPU对照应固定模型、失败task/episode、n_envs和RGB选项：先在原失败卡上单独运行，再让另一张卡同时运行；不要一边改卡数一边改n_envs或seed。若目标卡单独即失败，优先检查设备/映射；仅并发时失败再查整机压力和驱动；只随前序task变化则查上下文生命周期。此段是诊断顺序，不是已完成的远端多卡验收，也不要求重跑2500回合。

- `phase=reset`：尚未进入该回合策略rollout，优先查最终reset/相机渲染与缓存；final-frame step优化此时还未安装。
- `phase=simulation`且初始health非黑：初始化已通过，优先核对具体控制步的相机姿态、遮挡、最后一帧缓存和读回状态。
- 同一次渲染中用非零哨兵缓冲区复核：哨兵未被覆盖说明读回没有完成；已被完整覆盖为零则仍须区分有效黑画面、空场景/错误相机或绘制问题。不能仅靠再次重绘正常就当作原帧正确。

不要据此直接删除全黑检查，也不要把低速CPU渲染、关闭硬reset、更换seed等变化冒充相同测试协议。修复前先保留失败帧、原始异常及原环境；主对话负责诊断和改代码。

## 2026-10-29本地定位结果与边界

在SCO单H100、原固定seed上，针对WaffleReheat开展无模型的小诊断：原生渲染、共享渲染、deferred初始化＋final帧优化、16个并发环境均运行通过；同一worker还按WaffleReheat→CloseBlenderLid→WaffleReheat检查了前序环境影响。共27个短回合、81份三路观测数组均非黑；目标episode0在四种路径及前序环境切换前后的8次出现中，初始化和随后两个观测位置各自的三路RGB均逐字节一致。每回合只有初始化及两次零动作chunk，不能替代远端模型完整rollout或8卡验证。

这些短前缀的MjModel二进制序列化哈希均未变化，未观察到注册后模型资产改变；这不排除更长真实策略rollout。原始证据归档SHA256为`d12ae5866a34526c5cc010b00f978fde95b811a89a9c5a48eafd5c09adab2316`，测试MuJoCo3.3.1/驱动580.95.05，998项评测runtime manifest SHA256为`314afa30aa3016ab2ddf4a8801d2df676364952d96239b93ae158c619f22a105`。此次新增取证工具不改变这998项已封存运行文件。

另外用原生MuJoCo3.3.1渲染合法黑色平面，分别将输出缓冲区填为37/193，GPU读回都将全部分量覆盖为0，GL错误码0；现有观测检查仍抛出同样的All-zero错误。这个对照说明**全零检查不足以判定渲染故障**，不证明远端WaffleReheat就是合法遮挡。

静态检查未发现当前ScenePacket的Camera/Geom/Light结构布局与官方3.3.1头文件不符，或其scene内存所有权丢失。尚需排查的边界包括首次注册后model相关GPU资产是否变更、单路observable缓存是否刷新，以及原始读回是否写入。尚未定位远端根因，评测主路径和健康检查未因此改变。

上游参考：MuJoCo3.3.1的[EGL实现](https://github.com/google-deepmind/mujoco/blob/3.3.1/python/mujoco/egl/__init__.py)从EGL设备列表选择索引，不能靠CUDA逻辑编号推断物理配对。社区[robosuite #779](https://github.com/ARISE-Initiative/robosuite/issues/779)报告过受其它代码和reset影响的黑图/错图，但该报告使用mjviewer、MuJoCo3.4.0，与本项目固定路径不同，不能将其中关闭hard_reset等做法直接当作本问题修复。
