# FOCAL-VLA：RoboCasa365 全量 8 卡训练

本配方执行 **8 卡、global batch 64、70,000 个 optimizer steps，每 10,000 步存一次完整 checkpoint**。入口使用普通 `torchrun`。第一次从原始 OpenPI `pi05_base` 初始化；同一输出目录再次启动会恢复最新完整 checkpoint。

## 1. 固定训练配置

科学配置：[focalvla_robocasa365_8gpu_70k.json](../configs/focalvla_robocasa365_8gpu_70k.json)。

| 项目 | 本轮配置 |
|---|---|
| 数据 | 官方既定 scope 的 300 任务 × 每任务 100 条，共 30,000 条演示 |
| GPU | 单机 8 × H100 80GB；实体 8 卡吞吐待实际运行测量 |
| Batch | 每卡 8 × 8 卡 × 累积 1 = global 64 |
| 总训练量 | 70,000 updates；4,480,000 次样本抽取 |
| 初始化 | 原始 `pi05_base_pytorch/model.safetensors`；不用工程短训权重 |
| 学习率 | 前 10k 步线性 warmup 到 5e-5；之后 cosine，到第 70k 步为 5e-6 |
| AdamW | fused；betas=(0.9,0.95)，eps=1e-8，weight decay=1e-10 |
| 梯度 | clip norm=1；FP32 参数/Adam 状态，BF16 前向；无 EMA |
| 四项损失 | action + 0.05 geometry + 0.05 motion + 0.01 semantic |
| 保存 | 10k、20k、30k、40k、50k、60k、70k；模型、优化器、scheduler step、各 rank RNG、数据游标一起保存 |
| 随机种子 | 42；训练采样计划固定，各 rank 的动作噪声随机流独立且可恢复 |

论文 RoboCasa 原配方为 30k/global128；本轮按用户要求改成 70k/global64，cosine 终点随总步数延到 70k。网络、教师特征、分支可见性、损失形式与权重保留论文方法。离线标注增强、数据采样与轨迹末端处理见下文，避免把迁移设置混称为逐项相同的论文配方。

## 2. 资源和环境

在仓库根目录工作。复制 [资源模板](../configs/focalvla_resources.example.json) 为自己的 `resources.json`，填服务器上的绝对路径。这个文件只绑定资源位置，不修改训练超参数。

| 资源字段 | 所需内容 |
|---|---|
| `pi05_weights` | 原始 π0.5 base 的 PyTorch safetensors 文件 |
| `norm_assets` | RoboCasa365 同一 action/state 顺序的 `norm_stats.json` 所在目录；这只提供归一化统计，不提供微调权重 |
| `tokenizer` | 原生 `paligemma_tokenizer.model` |
| `dataset_root` | 含 `robocasa365-pretrain-atomic`、`robocasa365-pretrain-composite` 的 LeRobot v3 数据根；meta/episodes、meta/modality.json、parquet、三路视频齐全 |
| `training_scope` | 既定 official100 清单；`selected_episodes` 每条含 `dataset_name, episode_index, task_name`；须恰好 300×100，无重复 |
| `source_root` | 录制 extras 根；拼接方式为 `<source_root>/<source_prefix>/lerobot/extras/episode_XXXXXX/`，每集有 `states.npz, model.xml.gz, ep_meta.json` |
| `asset_roots` | robocasa、robosuite 各自 `models/assets` 的根目录，用于离线场景回放 |
| 教师代码和权重 | 模板其余六项；VGGT-1B、Track4World DA3 和 DA3 nested 权重，固定版本见 [teachers.json](../runtimes/focalvla/teachers.json) |

已有的匹配资源可以直接复用。π0.5 需要本包附带的三相机 OpenPI 实现和 Transformers 兼容补丁，不能只安装任意最新版 Transformers。

推荐两个 Python 3.11 环境：`policy` 用 Transformers 4.53.2，`prepare` 用 Track4World 所需的 4.57.3。两边的 PyTorch 均按已测版本安装：

```bash
# 在各自 venv 中执行；已有匹配环境可跳过。
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

# policy 环境
python -m pip install -r runtimes/focalvla/requirements-policy.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/focalvla/install_transformers_patch.py

# prepare 环境
python -m pip install -r runtimes/focalvla/requirements-prepare.txt
```

离线渲染用 OSMesa；Linux 主机需要 `libosmesa6`。FlashAttention 编译需要与 PyTorch 匹配的 CUDA 编译环境；也可复用已经可导入的 2.8.3 安装。依赖文件固定关键版本；本次实测使用既有环境，未把全新机器从零安装当成已完成的验收。

缺少教师资源时，在 `prepare` 环境执行：

```bash
python scripts/focalvla/provision_teachers.py --output /resources/focal_teachers
```

脚本从固定 Git commit 获取源码、从固定 Hugging Face revision 获取权重并校验权重 SHA256。把输出的 `teacher_resources.json` 六项填入自己的 `resources.json`。Track4World 移除未使用的 Pi3/Pi3X 顶层导入，并跳过会被完整 checkpoint 覆盖的 ConvNeXt 初始化权重下载；DA3 的计算与最终权重不变。

## 3. 一次性准备全量数据

以下在 `prepare` 环境中执行，`PREPARED` 使用容量足够的 SSD/高速共享存储。教师计算不在训练循环内运行。

```bash
export PYTHONPATH="$PWD/runtimes/focalvla${PYTHONPATH:+:$PYTHONPATH}"
export RESOURCES=/path/to/resources.json
export PREPARED=/path/to/focalvla_prepared_70k_gb64
export TORCH_HOME="$PREPARED/download_cache/torch"
export HF_HOME="$PREPARED/download_cache/huggingface"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
CONFIG=configs/focalvla_robocasa365_8gpu_70k.json

python -m focalvla.prepare plan --config "$CONFIG" \
  --resources "$RESOURCES" --prepared "$PREPARED"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/focalvla/prepare_fast.py --resources "$RESOURCES" \
  --prepared "$PREPARED" --status-dir "$PREPARED/progress" \
  --batch-size 8 --renderer osmesa --annotation-workers 4
python -m focalvla.prepare seal --config "$CONFIG" \
  --resources "$RESOURCES" --prepared "$PREPARED"
```

三个入口依次完成计划、标注与教师提取、封存；具体输出与含义：

1. `plan` 校验完整 scope，生成 70k×64 的确定性取样计划，并对实际用到的 `(episode,frame)` 去重。真实全量检查产生 **4,076,006 个唯一时刻**；全量数据有 4 条演示不足 50 帧。所有 30,000 条演示参与采样，未使用先前“每任务一条、每子任务一个 anchor”的短训样本集。任务概率正比于任务帧数的 0.4 次方，任务内按帧均匀采样；这属于本 RoboCasa365 迁移的固定数据协议。
2. `prepare_fast.py` 的每卡四个CPU进程提前标注后续演示，生成粗子任务文本、连续帧区间和实体名单；已有标注直接复用。当前默认是已完成300类抽查的v5标注器：人工语义模板结合录制的抓取、运动、接触、支持关系进行离线对齐。它是此前为修复小物体/遮挡误判采用的标注增强；**不是声称与论文Qwen3-VL-8B标注器逐项一致**。模型训练和推理不接收这些模拟器状态。覆盖连续性、实体有效性检查必须通过；自动检查不冒充人工逐条审核。独立执行`python -m focalvla.prepare annotate --config "$CONFIG" --resources "$RESOURCES" --prepared "$PREPARED" --workers 16`仍可汇总`ANNOTATION_QC.json`，但不必先串行等待全部标注完成。
3. `cache` 按 episode 分配到 8 张卡，各卡教师常驻，只计算计划实际需要的时刻。VGGT 看同步三路完整图像；Track4World 看完整 agent-left 的当前和未来 10 帧。相关实体 mask 只用于教师输出的池化。轨迹内部取 `t,t+5,…,t+50`，跨子任务边界不截断；episode 结束后使用最后一帧/最后一条动作补齐，既不跨 episode 也不丢弃短演示。论文未规定 episode 末端规则，这里显式固定。**缓存只保存数值状态/动作、目标特征、子任务和有效标志，不生成 PNG 或 `images.bin`。** 提取教师目标时的视频顺序解码、最多51帧缓冲只存在于内存。训练时从原视频按需读取当前三路图像。完成的 episode 可以复用，半成品会重算该集。
4. `seal` 汇总全量缓存、按实际训练抽样频次计算两类目标的均值/标准差，生成 `READY.json`。没有完整数据和目标就不会进入正式训练。规范化不混入评测数据。`geo_valid/motion_valid` 为 false 的样本只关闭对应辅助监督，action 和 semantic 仍参与训练。

完整的 407 万目标缓存需要独立的教师计算时间和磁盘容量；当前已完成的是全量采样计划与小规模端到端验收，**尚未生成这套全量缓存**。不要用已有 665 个短训 anchor 缓存替代它。改标注、种子、步数、batch 或教师版本时，使用新的 `PREPARED` 目录，重新生成与该计划对应的缓存。

优化入口只执行最终目标依赖的教师分支：VGGT最终聚合特征，以及DA3主干→Track4World 3D聚合特征；深度/相机预测头、metric模型、几何后处理和2D flow不参与目标，已跳过。保留三路518×518当前图像、11路630×630运动图像、原参考视角选择、原注意力及混合精度。参考视角选择按每条clip单独计算，避免批量小矩阵舍入影响离散argmin。教师常驻GPU，以8个目标合批；CPU进程顺序解码并用原OSMesa生成mask，最多预取两批，与GPU计算重叠。帧缓冲和队列均有上限，不保存RGB副本。

首次调用会生成`CACHE_IDENTITY.json`；同一计划/权重的完整旧缓存可继续使用。优化脚本单独记录SHA到每集`COMPLETE.json.execution`；不修改科学运行时代码来伪装缓存身份。恢复直接重跑同一命令；完整集跳过，尺寸不完整的集重算。`prepare_rank*.json`显示当前集与帧进度，`cache_performance_rank*.jsonl`记录逐集总时间、教师时间和输入等待；以去掉首集加载影响后的多集窗口衡量吞吐。`--episode-limit 2`仅用于隔离目录中的有界验收，正式全量命令不要添加。

按4,076,006个唯一时刻计算，FP32几何/运动目标合计约37.6 GB；加状态、动作、索引和标注为40多GB量级，不再为全量RGB另外预算TB级空间。原始MP4需要在整个训练期间可读。旧缓存中的教师数组也可以供新读取器使用；已有的PNG副本不会被自动删除。

`plan`完成后即可独立测试原视频供给，无需等待教师目标生成：

```bash
python scripts/focalvla/benchmark_video.py --resources "$RESOURCES" \
  --prepared "$PREPARED" --workers 4 --batches 96
# 32 workers用于整节点聚合压力测试；不是在一张卡的正式训练中开32 workers。
python scripts/focalvla/benchmark_video.py --resources "$RESOURCES" \
  --prepared "$PREPARED" --workers 32 --batches 256
```

该工具使用训练计划的真实随机视频样本和正式读取器，只测解码/拼批的持续供给；不包含模型计算、目标读取及原生归一化。2026-11-15在22 CPU核配额、初始`prefetch_factor=2`下，4 workers实测24.17个三视角样本/秒，32 workers实测103.15个/秒。后续完整训练A/B将正式配置调整为`prefetch_factor=4`；工具默认也为4，可显式传2复查初始配置。完整8卡训练步速需要在训练中确认，不能直接用该供给量替代。

## 4. 启动与恢复训练

切换到 `policy` 环境，保持 `RESOURCES`、`PREPARED` 指向上述文件和目录：

```bash
cd /path/to/robot-experiments
export RESOURCES=/path/to/resources.json
export PREPARED=/path/to/focalvla_prepared_70k_gb64
export OUTPUT=/path/to/runs/focalvla_70k_gb64
bash scripts/focalvla/train_8gpu.sh "$RESOURCES" "$PREPARED" "$OUTPUT"
```

恢复时执行**同一条命令**。不传 `--resume`，也不改输出目录。程序检查配置、源码、准备数据和 base 权重身份，加载最新完整 checkpoint，恢复 optimizer、学习率步数、每卡 RNG 和下一 global batch。`.tmp` 或尺寸不完整的 checkpoint 会跳过；发现不兼容的已完成 checkpoint 会报错。首次保存前中断时，只能重新从 base 开始，最多重做未保存的 10k 步。

`metrics.jsonl` 每10步记录四项损失、学习率、秒/步、samples/s、rank0数据等待、各rank中最大数据等待（`max_rank_data_wait_seconds_per_step`）和峰值显存。看最大rank等待可以发现其他卡供给不足而rank0正常的情况。旧尝试中超出恢复点的日志会归档，避免把回滚前后拼成虚假的连续进度。保存期间不计入常规速度窗口。每个完整checkpoint约41 GB，保留七个约需290 GB，另留写临时文件的空间。

DataLoader每卡4个worker，每个worker预取4个batch（每卡最多16个在途batch），持续准备GPU即将使用的数据。worker完成视频解码和原生预处理，主进程使用pin memory及非阻塞传输。`video.py`在一个batch内按视频分组、按时间排序解码，再还原样本顺序；同一帧不重复解码，不改变训练抽样和断点数据游标。

每个worker最多保留16个解码器、64 MiB的已请求RGB帧缓存。相邻请求最多顺序跨32帧，较远请求定位到前一个关键帧并按PTS解码到准确时刻；未请求的中间帧不转换成RGB。FFmpeg每个解码器只用1个线程，避免32个worker各自展开大量线程。8卡的RGB缓存总预算是2 GiB，另加固定数量的预取batch及解码器缓冲；缓存不会随训练增长，也不写磁盘副本。上述限制可通过科学配置末尾的`video_*`字段调整。

模型保留已测的共享context、五类可见性分块FlashAttention、三相机ViT合批、单层checkpoint和fused AdamW。实际8卡的磁盘带宽及解码吞吐仍需结合训练日志的数据等待时间测量；增加预取深度不能替代持续的解码吞吐。

单H100、B8完整训练对照已完成：224条真实样本覆盖158任务，模型/Adam初始状态、逐步随机数、样本顺序均一致；每组28步，排除前4步后统计24步，包含预处理、H2D、四项损失前后向、梯度裁剪和fused AdamW更新。初始2批预取出现一次5.619秒数据等待，均值比PNG慢37.55%，没有按这个结果发布。改为4批预取并交换测试顺序后，PNG为0.63157秒/步，视频为0.64320秒/步，平均差1.84%；中位数差1.29%，视频平均数据等待0.224毫秒、最大0.420毫秒，两组逐步损失最大差为0。此结果支持当前有界预取方案；未把单卡测试当成实体8卡验收。完整记录见[训练I/O对照](../experiments/exp008_focalvla/video_training_ab_20261115.json)。

## 5. 代码入口和验证边界

| 文件 | 职责 |
|---|---|
| `runtimes/focalvla/focalvla/model.py`、`attention.py` | FOCAL 训练前向、共享上下文与分块 FlashAttention |
| `reference.py` | 可读的可见性参考实现、保留 query 条件的动作推理 |
| `adapter.py`、`vendor/openpi/` | 原生三视角 π0.5 网络、tokenizer、归一化与动作接口 |
| `plan.py`、`prepare.py`、`teachers.py`、`annotation/` | 全量取样、离线子任务标注、教师目标和缓存封存 |
| `data.py`、`video.py` | mmap目标数组、原视频批量解码、有界内存缓存和固定global batch分片 |
| `train.py`、`checkpoint.py` | DDP 优化、日志、完整保存与恢复 |

论文的 8 geometry queries、8 motion queries、2048/256 维目标、MSE/SmoothL1(beta=1) 和原生 autoregressive semantic CE 都保留。semantic 标签只对 context+自身因果前缀可见；context、geometry、motion 和 action 均看不到 GT semantic tokens。动作专家读取当前 context 和两组 query latents。推理无需教师、未来图像、mask 或 GT 子任务文本。

CPU 协议测试：

```bash
export PYTHONPATH="$PWD/runtimes/focalvla${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s runtimes/focalvla/tests -v
```

已测范围与本次 portable GPU 验收记录见 [EXP-008](../experiments/exp008_focalvla/README.zh-CN.md)。此前两卡 global128 的 50 步短训耗时 4.308 秒/步、相对旧实现约 4 倍加速；该数字不作为本轮 8 卡 global64 的实测速度。70k 训练及策略成功率尚未运行。


## 多任务并行生成教师缓存

可将同一全量计划拆成5个独立双卡任务，共10卡。所有任务使用相同`RESOURCES`、`PREPARED`，各自设置不同的`SHARD_INDEX`（0到4）。以下命令分别在五个任务中执行；`--num-shards`和本地GPU数必须一致。

```bash
SHARD_INDEX=0  # 五个任务分别填写0、1、2、3、4
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/focalvla/prepare_fast.py --resources "$RESOURCES" \
  --prepared "$PREPARED" --status-dir "$PREPARED/progress/shard_$SHARD_INDEX" \
  --shard-index "$SHARD_INDEX" --num-shards 5 \
  --batch-size 8 --renderer osmesa --annotation-workers 4
```

全局worker编号为`shard_index × 2 + local_rank`，按10个worker轮转分配演示。30,000集正好每任务6,000集、每卡3,000集；不重叠、不漏集。完整旧缓存直接复用，不完整集由新所属worker重算。每个任务的日志与进度目录独立；缓存与只读计划共享。分片不改变教师、目标、取样计划或原GB64/70k训练配方，也不会训练出五个独立策略。原先覆盖同一缓存的未分片准备任务必须先退出。

五份任务全部完成后，执行一次前述`seal`，再使用原训练入口；不能在仅部分分片完成时启动训练。分片校验可运行`python scripts/focalvla/test_cache_partition.py -v`。普通单机入口不加分片参数，默认行为保持原样。

