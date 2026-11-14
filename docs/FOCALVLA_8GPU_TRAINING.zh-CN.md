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
python -m focalvla.prepare annotate --config "$CONFIG" \
  --resources "$RESOURCES" --prepared "$PREPARED" --workers 16
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m focalvla.prepare cache --config "$CONFIG" \
  --resources "$RESOURCES" --prepared "$PREPARED"
python -m focalvla.prepare seal --config "$CONFIG" \
  --resources "$RESOURCES" --prepared "$PREPARED"
```

四步的输出与含义：

1. `plan` 校验完整 scope，生成 70k×64 的确定性取样计划，并对实际用到的 `(episode,frame)` 去重。真实全量检查产生 **4,076,006 个唯一时刻**；全量数据有 4 条演示不足 50 帧。所有 30,000 条演示参与采样，未使用先前“每任务一条、每子任务一个 anchor”的短训样本集。任务概率正比于任务帧数的 0.4 次方，任务内按帧均匀采样；这属于本 RoboCasa365 迁移的固定数据协议。
2. `annotate` 对全部演示生成粗子任务文本、连续帧区间和实体名单。当前默认是已完成 300 类抽查的 v5 标注器：人工语义模板结合录制的抓取、运动、接触、支持关系进行离线对齐。它是此前为修复小物体/遮挡误判采用的标注增强；**不是声称与论文 Qwen3-VL-8B 标注器逐项一致**。模型训练和推理不接收这些模拟器状态。`ANNOTATION_QC.json` 列出失败记录；修正相应 `annotations/physics_v5/<episode>/LABELS.json` 后再重跑。覆盖连续性、实体有效性检查必须通过；自动检查不冒充人工逐条审核。
3. `cache` 按 episode 分配到 8 张卡，各卡教师常驻，只计算计划实际需要的时刻。VGGT 看同步三路完整图像；Track4World 看完整 agent-left 的当前和未来 10 帧。相关实体 mask 只用于教师输出的池化。轨迹内部取 `t,t+5,…,t+50`，跨子任务边界不截断；episode 结束后使用最后一帧/最后一条动作补齐，既不跨 episode 也不丢弃短演示。论文未规定 episode 末端规则，这里显式固定。缓存包含无损 RGB、数值状态/动作、目标特征和有效标志；每集一组文件，视频顺序解码，缓冲最多 51 帧。完成的 episode 可以复用，半成品会重算该集。
4. `seal` 汇总全量缓存、按实际训练抽样频次计算两类目标的均值/标准差，生成 `READY.json`。没有完整数据和目标就不会进入正式训练。规范化不混入评测数据。`geo_valid/motion_valid` 为 false 的样本只关闭对应辅助监督，action 和 semantic 仍参与训练。

完整的 407 万目标缓存需要独立的教师计算时间和磁盘容量；当前已完成的是全量采样计划与小规模端到端验收，**尚未生成这套全量缓存**。不要用已有 665 个短训 anchor 缓存替代它。改标注、种子、步数、batch 或教师版本时，使用新的 `PREPARED` 目录，重新生成与该计划对应的缓存。

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

`metrics.jsonl` 每 10 步记录四项损失、学习率、秒/步、samples/s、rank0 数据等待和峰值显存。旧尝试中超出恢复点的日志会归档，避免把回滚前后拼成虚假的连续进度。保存期间不计入常规速度窗口。每个完整 checkpoint 约 41 GB，保留七个约需 290 GB，另留写临时文件的空间。

DataLoader 每卡 4 个 worker、pin memory、预取 2 个 batch，有界 episode 缓存；训练不随机 seek 长视频。模型保留已测的共享 context、五类可见性分块 FlashAttention、三相机 ViT 合批、单层 checkpoint 和 fused AdamW。没有整模型外层重复 checkpoint，也没有随训练增长的样本缓存。

## 5. 代码入口和验证边界

| 文件 | 职责 |
|---|---|
| `runtimes/focalvla/focalvla/model.py`、`attention.py` | FOCAL 训练前向、共享上下文与分块 FlashAttention |
| `reference.py` | 可读的可见性参考实现、保留 query 条件的动作推理 |
| `adapter.py`、`vendor/openpi/` | 原生三视角 π0.5 网络、tokenizer、归一化与动作接口 |
| `plan.py`、`prepare.py`、`teachers.py`、`annotation/` | 全量取样、离线子任务标注、教师目标和缓存封存 |
| `data.py` | mmap 数组、无损 RGB 读取和固定 global batch 分片 |
| `train.py`、`checkpoint.py` | DDP 优化、日志、完整保存与恢复 |

论文的 8 geometry queries、8 motion queries、2048/256 维目标、MSE/SmoothL1(beta=1) 和原生 autoregressive semantic CE 都保留。semantic 标签只对 context+自身因果前缀可见；context、geometry、motion 和 action 均看不到 GT semantic tokens。动作专家读取当前 context 和两组 query latents。推理无需教师、未来图像、mask 或 GT 子任务文本。

CPU 协议测试：

```bash
export PYTHONPATH="$PWD/runtimes/focalvla${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s runtimes/focalvla/tests -v
```

已测范围与本次 portable GPU 验收记录见 [EXP-008](../experiments/exp008_focalvla/README.zh-CN.md)。此前两卡 global128 的 50 步短训耗时 4.308 秒/步、相对旧实现约 4 倍加速；该数字不作为本轮 8 卡 global64 的实测速度。70k 训练及策略成功率尚未运行。
