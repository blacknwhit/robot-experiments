# 新集群交接：PI0.5、最佳 Query-FIXED、3D trajectory v2

本手册针对 Linux + NVIDIA CUDA 集群。所有命令从仓库根目录执行，除非明确说明。调度器可以是 Slurm、Kubernetes 或其它系统；代码不依赖原 SCO 平台、原登录主机或旧目录。当前包没有为目标集群提交训练或评测任务。

## 1. 必须先理解的复现边界

本仓库包含四个入口、三套独立模型源码。`baseline` 的历史源码与审计快照匹配；`query_fixed` 使用可核对的完整历史树重构，原训练主仓 commit 未能从本地归档恢复，但主要源码和正式评测源文件有逐项哈希证据；3D v2 使用实际部署的模型、数据 reader、loss 和 trainer。

本次新增的是路径配置、安装/下载、同卡数恢复选择和验收、启动锁以及可移植评测包装。未改科学目标、模型结构、Adam/LR 或训练样本筛选。完整列表见 `provenance/PORTABILITY.md`。新包已做的本机检查见 `provenance/VALIDATION.json`；**本机 CPU 检查不代表新集群的 GPU、NCCL、EGL、训练恢复或 benchmark 已验收**。必须执行下方 smoke。

已有分数来自旧集群：baseline 391、FIXED 559、Query3D-v2 373、Embedding3D-v2 443，分母均为2500。历史 robosuite 精确 Git SHA 未留存；本包选择官方要求的源码分支上的固定 commit，需记录新集群环境，不应宣称跨集群逐位复现这些分数。

## 2. 目录、硬件与存储

```bash
git clone https://github.com/blacknwhit/robot-experiments.git
cd robot-experiments
export ROBOT_EXPERIMENTS_HOME=/your/absolute/shared/robot-data
mkdir -p "$ROBOT_EXPERIMENTS_HOME"
export H="$ROBOT_EXPERIMENTS_HOME"
```

将 `/your/absolute/shared/robot-data` 换成实际路径。代码目录与数据目录分开。尽量用本地 SSD 缓存视频/Arrow，checkpoint 用可靠共享存储。目录名建议不包含空格。

训练支持单机 2、4、8 GPU，micro batch 分别32、16、8，accumulation=1，global batch=64。旧训练用过2×H100和8×RTX5090；不要只凭更多卡推定更快。**baseline 历史 GC=false，建议先用历史8卡B8做复现**；2卡B32的显存是否足够必须实测，不自动改batch或打开GC。FIXED与3D的GC=true。A100等其它硬件需先验收，本包没有给它们性能承诺。

运行环境：Python3.11、PyTorch2.7.1+CUDA12.8、Transformers4.53.2且必须打对应版本patch；驱动需支持CUDA12.8。用管理员/集群提供的CUDA GPU镜像和NVIDIA驱动。不要在CPU登录节点执行训练。EGL需要图形驱动库，CUDA tensor可用不代表仿真可渲染。

空间规划应包含两个完整公开视频数据分区、原始extras、Arrow缓存和GT。GT数组历史大小350,517,277,868 bytes，另有源文件/索引。每个完整训练checkpoint含模型、Adam FP32、主权重、RNG等，量级为数十GB；每1000步保存且本包**不自动删旧checkpoint**，完整70k运行可能占数TB。开始前按实际首个checkpoint大小预算；自行归档旧完整checkpoint时保留最新多个和70k里程碑，不能删正在写入的目录。

## 3. 安装两个环境

先在可联网安装节点准备Git、C/C++工具链和uv：

```bash
python3 -m pip install uv
bash scripts/setup_policy.sh
bash scripts/setup_sim.sh
export PY="$H/envs/policy/bin/python"
export SIM_PY="$H/envs/sim/bin/python"
```

`setup_policy.sh` 从保留的OpenPI `uv.lock`开始，再明确安装训练依赖和CUDA12.8 wheel；它是**历史锁 + 显式兼容覆盖**，不是声称整个旧容器已有完整lock。NCCL2.26.5是对Torch依赖pin的有意覆盖；保留 `receipts/policy.freeze.txt`。如果解析或wheel安装失败，先修复具体依赖，不运行半安装环境。重新运行 `uv sync` 会把覆盖还原，应重新完整运行setup脚本。

三个runtime各有隔离的 `runtime/transformers` overlay。不要把patch复制到全局site-packages，也不要在同一Python进程切换runtime。`scripts/*`会选择正确PYTHONPATH。训练/策略进程使用轻量RoboCasa shim来满足OpenPI eager imports；仿真进程使用真实RoboCasa，不能带shim。

simulator 固定：RoboCasa v1.0 `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`，robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`，NumPy2.2.5、MuJoCo3.3.1。policy/GT用NumPy1.26.4，两环境不要混装。安装器下载厨房资产并创建macros，检查 `$H/playground/Code/robocasa365/robocasa/macros_private.py`。配置路径需要指向已下载资产；不能复制旧集群macros。

```bash
for V in baseline query_fixed query3d_v2 embedding3d_v2; do
  "$PY" scripts/doctor.py --variant "$V"
done
# 以下在已分配GPU的任务中执行：
"$PY" scripts/doctor.py --variant query_fixed --cuda
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/nccl_probe.py
```

8卡验收将nproc改8；检查每rank输出、GPU型号和NCCL sum。`doctor`失败不能被忽略。

## 4. 下载数据与初始化资源

```bash
"$PY" scripts/download.py --home "$H" --component small
"$PY" scripts/download.py --home "$H" --component datasets
"$PY" scripts/download.py --home "$H" --component base
"$PY" scripts/convert_base.py --home "$H"
```

`datasets`下载两个公开flat LeRobot v3聚合分区，含视频，体积很大。断网后可重运行，HuggingFace负责缓存/续传；若源站需要登录，使用自己的HF凭据，勿写进代码。固定revision：

| 来源 | 固定版本 |
|---|---|
| [atomic](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | `d5eac857ee9b07ff089e89c594c64493b8df6e41` |
| [composite](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | `4a5e4552f8823b8c92944e08bba1ff268c51bef9` |
| 3D原始状态：[NVIDIA Kitchen Demos](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos) | `522e4ffa3bb9d9729f79b241c65fc9e589f2659c` |
| [RoboCasa norm](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/916d8ec4aab4e6f930ba6b5b9bb8537ad16d4bb8) | `916d8ec4aab4e6f930ba6b5b9bb8537ad16d4bb8`，只取75k的assets/norm_stats.json |

**基座是 `gs://openpi-assets/checkpoints/pi05_base`，不是RoboCasa已训练75k参数。** 下载器用公开GCS接口、记录object generation；转换为BF16 PyTorch后核对历史SHA `9fd36e337832960bbfc122bd686dfc64b89551522d891c0d21ae586385de297e`。转换需较大CPU内存，建议在资源足够的CPU任务中做。转换SHA不同会停下，先检查源对象、转换工具/精度和tensor身份，不跳过校验冒充旧base。底座/数据仍遵守各上游使用条款。

norm的SHA必须是 `4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee`。Tokenizer下载到 `$H/cache/openpi/big_vision/paligemma_tokenizer.model`，文件来源和SHA记入下载receipt；训练不需要旧集群cache。

不要把32,043条聚合演示全部用于训练。训练严格从300个task各取100条：按source episode排序，再seed0 shuffle，任务权重为长度的0.4次方。保留source_prefix/source_episode_index和原parquet列，不换任意重新转换的数据。

```bash
"$PY" scripts/data_scope.py --home "$H"
"$PY" scripts/prepare_arrow.py --home "$H"
```

必须得到300 tasks / 30,000 episodes / 27,815,381 frames。训练registry单独冻结为SHA `23e8e31bb56bae80de86c34eafadfe32dd1e1eba6608376152ba9fdda5532a4b`。**不要覆盖simulator的registry**，两者horizon定义有历史差异。本轮只验证公开镜像episode metadata SHA与旧记录一致，没有重新下载逐文件核对全集视频。

## 5. 3D两版额外生成GT

基线、FIXED跳过本节。3D v2使用机械臂/直接操作对象表面点，1024点、无背景、米制、窗口首帧机器人base坐标；不是旧contact-patch或随便挑运动点。

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage prepare
# 可提交8个CPU任务，分别替换 I=0..7；不能多个任务使用同一个shard-index。
I=0
"$PY" scripts/prepare_gt.py --home "$H" --stage generate \
  --workers 6 --shard-index "$I" --num-shards 8
# 8个分片全部结束后：
"$PY" scripts/prepare_gt.py --home "$H" --stage verify
```

如不分片，省略shard参数默认一个任务处理全集。源下载器按 `source_prefix/lerobot/extras/episode_NNNNNN/{states.npz,model.xml.gz,ep_meta.json}` 和对应原始episode parquet下载，并核对当前/原始 state、action、frame_index、timestamp。通过记录的MuJoCo状态重放，不积分动作。资产预期在 `$H/playground/Code/{robocasa365,robosuite}/.../models/assets`，第3节已建立。

`asset_root`属于GT配置哈希。移到新根后，脚本计算新config SHA；`configure.py`读取它，不能硬填旧 `04333d...`。源代码路径去除与重新生成会改变provenance哈希，这是新的可追溯生成，而不是旧GT文件的逐字节复制。verify逐episode核对源身份、shape、数组SHA及范围，成功写 `PORTABLE_GT_ACCEPTED.json`。正式3D训练要求此标记。

历史分布供排错：29,812条为512机械臂+512对象，188条为1024机械臂，背景0。旧GT数组约350.5GB。若新生成出现大量无对象/几何语义未知、帧数差异或重放失败，先检查资产版本、XML和原始状态，不静默忽略演示。

## 6. 真实训练/保存/恢复 smoke

先在已分配的GPU任务中执行。例子为Query3D-v2、2×H100；替换V可检查另外三版，baseline优先N=8。不要把smoke权重用于正式训练。

```bash
V=query3d_v2
N=2
mkdir -p resolved
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-smoke" \
  --gpus "$N" --smoke --stop 2 --out "resolved/${V}-smoke2.yaml"
"$PY" scripts/train.py --config "resolved/${V}-smoke2.yaml"
# 新进程恢复完整step2，执行step3；只改变stop gate，保持LR100k日程：
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-smoke" \
  --gpus "$N" --smoke --stop 3 --out "resolved/${V}-smoke3.yaml"
"$PY" scripts/train.py --config "resolved/${V}-smoke3.yaml" --resume
```

检查step1/2/3均有完整checkpoint和PORTABLE_COMPLETE；每rank恢复审计应报告FP32主权重/Adam、BF16和RNG一致。FIXED必须额外有 `custom_checkpoint_2.pkl` 教师状态（decay1.0仍有时钟），两版3D必须有trajectory_contract。loss/梯度应有限，Query/点头/Embedding和共享VLM梯度/FP32参数确实更新。训练日志已有相应诊断；不能只看RUNNING。

本包仅支持**相同world size、相同配方的完整续训**。2→8需要单独审计过的ZeRO重分片，本包没有把它伪装成自动支持。完整恢复选择会拒绝缺失Adam/RNG、错误batch/world、时钟或v2contract的checkpoint，并向前查找完整状态；无完整状态则报错，不能再base重跑。run目录第一次创建后再次fresh也会拒绝。新集群首次运行请保留smoke日志与freeze记录。

## 7. 正式训练到70k

```bash
V=query_fixed
N=2
"$PY" scripts/configure.py --variant "$V" --home "$H" --run-id "${V}-s42" \
  --gpus "$N" --out "resolved/${V}-formal.yaml"
"$PY" scripts/train.py --config "resolved/${V}-formal.yaml" --dry-run
"$PY" scripts/train.py --config "resolved/${V}-formal.yaml"
# 调度器下一次启动用：
"$PY" scripts/train.py --config "resolved/${V}-formal.yaml" --resume
```

对每版使用独立run-id。将命令放进集群GPU任务，不在登录宿主执行。launch持有writer锁；自动恢复由一个调度机制负责，不能同时跑多个恢复循环。SCO上仍按原约定native autoRetrain/10min；其他集群配置对应的单一重试入口，并在重试时传 `--resume`。发生代码错误应先停止重复失败并保留现场。

| 科学项 | baseline | Query-FIXED | Query3D-v2 | Embedding3D-v2 |
|---|---|---|---|---|
| sampler | IID | G8 ordered_random | G8 ordered_random | G8 ordered_random |
| Query | 无 | 64 prefix Query | 50 prefix Query | 不改prefix；训练期2层Embedding |
| 辅助监督 | 无 | offsets25/50 × 3相机，固定视觉教师 | 相邻位移+累计10/25/50步 | 同Query3D |
| 辅助权重 | 无 | 0.1 | 3.75 | 4.0 |
| 辅助warmup | 无 | 1000 update | 1000 update | 1000 update |
| LR warmup / horizon | 1000 / 100000 | 同左 | 同左 | 同左 |
| 梯度检查点 | false | true | true | true |

共有：base初始化、seed42、global64、acc1、AdamW β=(0.9,0.95)、eps1e-8、weight_decay1e-10、peakLR2.5e-5、minLR2.5e-6、clip1、raw model（无policy EMA）。**FIXED教师decay=1.0是固定初始教师，不是动态EMA教师。** 70k停止不把学习率日程改成70k。

除baseline每5k保存外，其余每1k保存；均增加first-update完整保存。正式至少检查最初3条有限loss/gradient、预热40步后两个不重叠100步速度窗口，以及首个1k完整状态（baseline按原5k保存配方）。不要用训练总loss当closed-loop分数。保留日志和各rank恢复审计。

## 8. 准备评测模型

新训练的70k评测权重：

```text
$H/runs/<run-id>/checkpoints/steps_70000_model.safetensors
```

同时保留run的config.full.yaml、config.yaml、dataset_statistics.json（若生成）、openpi_assets/norm_stats.json、portable_recipe.json和完整checkpoint。评测脚本显式接收已解析配置；模型全部key严格加载，包括tied aliases；不会把初始化允许缺key的逻辑用于已训练模型，也不全局cast BF16。

若搬运旧训练结果：用你自己的安全传输工具复制**整个run目录**，再单独重定位配置里的OpenPI、norm及数据路径；不要复制旧token、SCO提交回执或旧host宏。GitHub不提供旧私有训练参数的下载链接。旧FIXED70k权重SHA：`fa5c5078afb8efa807a7af6129ff969ab1c0c4ccebf9e10180bb18c2a1d855a2`。搬来只有safetensors时可做严格推理，但不能声称恢复optimizer/data clock。

如只想确认推理/仿真管线，也可对训练smoke的step3模型做下节smoke；这个验收绑定该checkpoint，正式70k模型需自己的smoke。

## 9. 先一片smoke，再全量：每task独占一张卡

固定50 task，原task顺序和已经乘1.5的horizon保存在 `assets/eval_tasks.json`，**不能再乘1.5**。seed7，50 trials/task，单env，replan5，10次去噪，EGL，严格eager、TF32关闭、stateless_request_v1噪声与逐episode trace。

```bash
V=query_fixed
CFG="$PWD/resolved/${V}-formal.yaml"
CKPT="$H/runs/${V}-s42/checkpoints/steps_70000_model.safetensors"
OUT="$H/evaluation/${V}-70k"
# 在GPU分片中运行，只有第一个task跑2个episode作为smoke：
"$PY" scripts/evaluate.py --variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT" \
  --sim-python "$SIM_PY" --output "$OUT" --part 1 --gpu-ids 0,1,2,3 --smoke
```

smoke不是看是否成功完成任务：需要strict完整模型加载、正确模型握手、有限动作、EGL仿真完成、两条有效且独立的trace。3D版本禁止点头在推理执行，输入不含3D点。成功写入 `SMOKE_ACCEPTED.json` 后可放行全量，不要求每片都重复smoke。

正式布局为12个4GPU任务 + 1个2GPU任务，一task/GPU。按新集群调度器分配；每个GPU任务在自己的可见GPU集合上执行：

```bash
# PART=1..12，每项申请4GPU；不是在同一台4卡机器同时启动12份。
PART=1
"$PY" scripts/evaluate.py --variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT" \
  --sim-python "$SIM_PY" --output "$OUT" --part "$PART" --gpu-ids 0,1,2,3
# PART=13申请2GPU：
"$PY" scripts/evaluate.py --variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT" \
  --sim-python "$SIM_PY" --output "$OUT" --part 13 --gpu-ids 0,1
```

一张可用debug卡也可smoke：`--smoke --gpu-ids 0`。分片可重运行：已完成结果必须identity匹配并再次验收才跳过，失败结果不会混入其它checkpoint。多个variant要用不同output，并用不同port-base或隔离网络命名空间。端口按part偏移，运行前检查占用，连接后校验模型SHA/variant/config；不凭TCP通了就承认模型正确。

**渲染绑卡与CPU线程：**策略server按`--gpu-ids`设置`CUDA_VISIBLE_DEVICES`；模拟器移除该CUDA变量，按对应EGL设备索引设置`MUJOCO_EGL_DEVICE_ID`，不会把每个模拟器都绑定到EGL 0。默认两类编号相同，例如`--gpu-ids 0,1`对应EGL 0、1。遇到Slurm、容器或UUID设备时，先在任务内核对CUDA与EGL所指GPU的UUID/PCI地址；如顺序不同，显式传`--gpu-ids 0,1 --egl-device-ids 2,5`。这里2、5仅为映射示例，不是可直接用于任意集群的编号。脚本不会自动验证物理GPU对应关系，`--dry-run`会显示两组选择。

两个子进程在导入数值库之前都强制设置`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`，恢复旧SCO启动限制；不会继承外部过大的线程池设置。该限制由公共environment构造器设置，也适用于使用它的训练子进程。部署到新集群仍需做两卡并行短测，核对实际渲染GPU及耗时；CPU环境传递测试不等于EGL验收或已测得加速。推理去噪循环、compile/TF32、seed和replan协议保持原样。

全片结束后汇总：

```bash
"$PY" scripts/evaluate.py --variant "$V" --home "$H" --config "$CFG" --checkpoint "$CKPT" \
  --sim-python "$SIM_PY" --output "$OUT" --summarize
```

汇总独立核验50×50 raw successes、2500个唯一trace、协议、模型/配置/环境identity，生成 `FINAL_ACCEPTED.json`。不要用文件数、任务SUCCEEDED或单个完成marker代替验收。`protocol_comparable=false`沿用旧脚本定义：它只对官方指定checkpoint精确复现设true；本实验仍使用相同benchmark协议，但不冒充官方权重复现。

## 10. 常见失败与交接产物

- **import错误**：先doctor，核对Python/PYTHONPATH、transformers overlay、pytorch3d，不把sim的NumPy2混到policy。
- **数据范围不一致**：检查revision、source身份、100_demos筛选；不按当前文件数猜训练范围。
- **GT缺资产/几何不识别**：检查固定代码版本、macros和原始XML；保留失败episode，不用零轨迹掩盖。
- **NCCL慢/卡住**：检查拓扑、实际PCIe速率、驱动、collective计时和当前attempt，不从8卡数推断吞吐；固定global64再比较。
- **恢复失败**：保留完整checkpoint，检查rank数、Adam、scheduler、RNG、data clocks、teacher/trajectory contract；禁止用最新safetensors替代完整恢复。
- **EGL失败**：检查任务内NVIDIA图形驱动挂载与设备映射；不要改成别的renderer后仍称同协议。
- **评测结果异常**：先比较正确checkpoint SHA、任务覆盖、seed/horizon/replan、raw数组和trace；不能只比较总平均。

在新集群保留：仓库commit、源码manifest、download receipts、两份freeze、GT验收、resolved configs、smoke完整日志/状态、训练日志和resume audits、70k完整state+推理权重、50份raw评测JSON和最终验收。复现成功以这些证据为准。

上游参考：[OpenPI](https://github.com/Physical-Intelligence/openpi)、[RoboCasa v1.0安装](https://github.com/robocasa/robocasa/blob/v1.0/docs/introduction/installation.md)、[数据与extras说明](https://github.com/robocasa/robocasa/blob/v1.0/docs/datasets/using_datasets.md)。
