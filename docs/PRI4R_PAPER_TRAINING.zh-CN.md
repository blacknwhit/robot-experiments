# PI0.5 + Pri4R：RoboCasa365训练与评测

本入口是Pri4R公开方法的独立实现，variant为`pri4r_paper`。π0.5本体对齐[官方PyTorch commit 215abfb](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)；主配方遵循[论文v2](https://arxiv.org/html/2603.01549v2)。作者没有公开完整训练实现，未披露的辅助结构/坐标/采样细节采用下文已明确选择的约定。

**当前版本已通过真实单H100短训、保存、新进程恢复、推理、官方前向/梯度对照及评测服务验证。** 四进程CPU归约和混合dtype状态恢复通过；每卡B32容量已用单H100的DDP＋GPU Adam/EMA实测。尚未四张实体GPU联跑，也未开始正式30k训练或完整benchmark，不承诺性能提升。

## 方法与配方

| 项目 | 固定设置 |
|---|---|
| 初始化/训练 | 原始`pi05_base`，全量微调；不使用历史70k或工程权重 |
| 原π0.5路径 | 官方selective BF16/FP32、原joint attention、原flow matching；原生DDP＋torch AdamW，无外层autocast |
| 辅助结构 | 独立50个query读取末层prefix；2层self→cross→FFN，d2048/8头/FFN4096；不插入动作prefix |
| 两MLP | Point 3→128→2048；Fusion 4096→256→3；GELU，无新增RoPE或时间因果模块 |
| GT | 首帧机器人中心、随其朝向的**边长3m**立方体内，按mesh表面积均匀采1024点；固定面片/重心坐标跨帧跟踪 |
| 坐标与目标 | **世界坐标、米**；50段相邻位移，不额外标准化、不加累计/绝对位置loss |
| 损失 | action FM + **1×有效XYZ逐元素平均L1**；缺失未来点mask，不用未来运动筛点或类别配额 |
| 正式训练 | 默认4×H10080GB，每卡B32、acc1、global128；30000优化步 |
| LR/AdamW | peak5e-5、warmup10000、cosine；未披露的末值用官方通用默认2.5e-6；beta(.9,.95)、eps1e-8、所有参数WD1e-10、clip1 |
| EMA | .999，每次optimizer update后更新全部唯一参数的FP32 shadow；正式CUDA，单卡短测CPU；同时保留raw和EMA |
| 保存 | 第1步额外完整保存，其后每1000步；原生模型、Adam、scheduler、每rank RNG、数据进度、EMA |
| 推理 | 严格加载完整EMA权重后删除辅助模块；只用原始图像/语言/状态输入，输出环境动作50×12 |

`mixed_precision=no`指关闭Accelerate的自动整体转换，不是全模型FP32。本次实际参数为FP32 221,874,339个、BF16 3,497,036,912个。原生Adam按参数dtype维护状态，没有ZeRO的FP32 master；旧ZeRO验收标准和checkpoint不能直接套用。

论文明确EMA=.999，但官方PyTorch trainer没有EMA实现；本仓库补充标准FP32 EMA，作者的内部细节仍未知。数据集仍是用户指定RoboCasa365 300任务×100示范，论文为24任务/Human-50，因此这里是方法迁移，不是论文原分数复现。判断增益时需对齐baseline的base、数据、训练预算与评测协议。

采点范围来自4条示范的有限诊断：2m漏掉初态夹爪，3m保留全部夹爪及约98.3%–99%的机器人表面，相比4m减少约37%–48%的可采面积。静态点仍占多数；没有据未来运动筛点，不能称3m为300任务的全局最优。详见[采样证据](../provenance/PRI4R_CUBE_DIAGNOSTIC.json)。GT数组的`stored_frame`是world；训练实际坐标由recipe及`point_meta.coordinate_frame`明确选择，历史metadata里的`read_frame`描述旧reader默认值。

## 1. 环境和资源

```bash
export ROBOT_EXPERIMENTS_HOME=/absolute/path/to/pri4r_resources
export PRI4R_ASSET_ROOT="$ROBOT_EXPERIMENTS_HOME"
export EXPERIMENT_ROOT=/absolute/path/to/pri4r_experiment
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"
export PY="$ROBOT_EXPERIMENTS_HOME/envs/policy/bin/python"
```

优先复用原有PI0.5/FIXED资源。资源根需要：`data/robocasa365/{robocasa365-pretrain-atomic,robocasa365-pretrain-composite}`、`models/pi05_base_pytorch/model.safetensors`、`models/robocasa_norm/norm_stats.json`、`cache/openpi`和`cache/arrow`。缺少资源按[主交接文档](../HANDOFF.zh-CN.md)的下载、base转换和环境配置步骤操作，数据及权重不在GitHub中。

```bash
bash scripts/setup_policy.sh
"$PY" scripts/runtime_overlay.py --variant pri4r_paper
```

已验收环境为Python3.11、Torch2.7.1+cu128、Transformers4.53.2、Accelerate1.5.2、NCCL2.26.5；安装脚本仍包含共享runtime所需的DeepSpeed依赖，但本训练入口不使用DeepSpeed引擎。不要统一调用model.bfloat16()或重新打开外层autocast。

GT需要RoboCasa/robosuite真实mesh资产。`PRI4R_ASSET_ROOT`是解析符号链接后的文件边界，必须包含源示范、mesh资产及GT输出；复用旧资源时可把新资源目录放在旧资产根内，并把该变量设为旧资产根。不要为了通过路径检查指向不相关目录。

## 2. 可选的少量GT与部署短测

新集群复用相同代码与兼容环境可直接准备全量训练；以下用于资源/环境变更后做有界验证。默认只生成4条示范，不会自动扩成全量。

```bash
"$PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage prepare
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage generate --workers 2
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage verify

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 1 --smoke --stop 2 \
  --out resolved/pri4r_smoke2.yaml
"$PY" scripts/import_preflight.py --config resolved/pri4r_smoke2.yaml
"$PY" scripts/train_or_resume.py --config resolved/pri4r_smoke2.yaml

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 1 --smoke --stop 3 \
  --out resolved/pri4r_smoke3.yaml
"$PY" scripts/train_or_resume.py --config resolved/pri4r_smoke3.yaml
"$PY" scripts/point_inference_probe.py --config resolved/pri4r_smoke3.yaml \
  --checkpoint "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/checkpoints/steps_3_model.safetensors" \
  --sample "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/engineering_sample.pt" \
  --output "$EXPERIMENT_ROOT/pri4r_inference_check.json"
```

单卡短测B4/acc32/global128，**Adam在GPU、EMA在CPU、数据worker=0**。这些设置保证单卡有界验证，其速度不代表正式四卡。短测仅使用已准备GT的真实示范；formal模式拒绝绑定短测清单。保存/恢复需要同一封存release、recipe和world，缺完整状态会报错，不静默回到base。

## 3. 远端全量GT

使用新的完整资源/GT目录，复用相同base/data/cache，保留短测产物。若资源根改变，同步更新`OPENPI_DATA_HOME`。下列命令只应在确实要正式训练时运行：

```bash
"$PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage prepare --full
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage generate --full \
  --workers 6 --num-shards 8 --shard-index 0
# 其余CPU作业使用shard-index 1..7，同一配置与manifest。
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage verify --full
```

完整范围要求30000示范/27815381帧，GT约350GB，另留源与缓存空间。每条源身份、配置和数组SHA均检查；完成证据是`data/gt_pri4r_paper/PORTABLE_GT_ACCEPTED.json`。不能用旧armobj配额标签、4m标签或旧完成marker冒充当前3m约定。

## 4. 四卡正式训练和完整续训

```bash
"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_paper_seed42 --gpus 4 --formal --stop 30000 \
  --out resolved/pri4r_formal.yaml
"$PY" scripts/train_or_resume.py --config resolved/pri4r_formal.yaml
```

入口自动使用torchrun启动4个rank，保持每卡B32、acc1、global128及CUDA EMA。数据loader默认每rank4个worker、prefetch2；不要套4份外层训练命令。SCO新提交按用户约定配置原生autoRetrain=true、10分钟检查；其它集群重启同一入口即可，不创建竞争writer。

重新启动同一条命令会选最新完整兼容checkpoint，恢复模型/Adam/scheduler/每rank RNG/数据进度/EMA。保持同一checkout和resolved配置；旧649dd36的ZeRO工程状态不迁入当前原生DDP。正式训练从原base开始，工程step3权重不用于正式初始化。

查看`point_track_metrics.jsonl`、`native_update_rank*.jsonl`（短测）、`native_restore_rank*.json`、`ema_restore_rank*.json`、`save_timing.jsonl`及checkpoint下的`PORTABLE_COMPLETE.json`。RNG恢复采用严格失败策略，不能把Accelerate的静默RNG加载失败当成功。

## 5. 训练后评测

使用EMA导出`steps_N_model.safetensors`，raw导出为`steps_N_raw_model.safetensors`，两者不要混淆。保持训练对应的resolved配置及归一化资产。当前服务端完整加载后移除辅助头，不需要3D GT。

```bash
SIM_PY=/absolute/path/to/simulator/python
CHECKPOINT="$EXPERIMENT_ROOT/runs/pri4r_paper_seed42/checkpoints/steps_30000_model.safetensors"
OUT="$EXPERIMENT_ROOT/eval/pri4r-30k-seeds-v1"
"$PY" scripts/evaluate_parallel.py \
  --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --config resolved/pri4r_formal.yaml --checkpoint "$CHECKPOINT" \
  --sim-python "$SIM_PY" --output "$OUT" --all-tasks \
  --gpu-ids 0,1,2,3,4,5,6,7 --egl-device-ids 0,1,2,3,4,5,6,7 \
  --n-envs 16 --tasks-per-gpu 50 --simulator-cpu-threads 1 \
  --rgb-schedule final --initial-rgb deferred --timeout 172800
```

评测GPU数量独立于训练：单卡/2卡/4卡相应缩短两组设备号。模型每卡一份，多task共享环境池；2500个task/episode种子固定，不生成场景计划、不做旧参考像素哈希比较。保留完整模型加载、实际请求eager/Graph对照、观测/动作健康及2500回合覆盖检查。模型迭代不改变测试任务、种子或horizon。

模拟器安装与`receipts/sim.freeze.txt`、容器NVIDIA_DRIVER_CAPABILITIES=all及GPU/EGL实际映射要求，沿用[Astra执行文档](ASTRA_FIXED_SCENES.zh-CN.md)。本版已验证实际服务端5个请求/4客户端并发及eager/Graph一致性；本次没有重新执行Pri4R的全50任务闭环，更没有30000步模型成绩。

## 实测与边界

完整原始汇总见[验证记录](../provenance/PRI4R_PAPER_VALIDATION.json)、[官方对照](../provenance/PRI4R_OFFICIAL_PARITY.json)、[辅助梯度](../provenance/PRI4R_AUXILIARY_GRADIENTS.json)和[容量探针](../provenance/PRI4R_CAPACITY_PROFILE.json)。

- 真实单H100：update1/2完整保存，新进程恢复2→3；模型/Adam/RNG及859个唯一参数的EMA恢复逐值一致。
- 同一真实输入、原base权重、noise/time及增强随机流：与官方动作前向逐值一致，**805个梯度张量逐值一致**。这是一个实样本对照，不是穷尽所有输入。
- 辅助loss单独反向传播时，18层VLM语言主干均有有限非零梯度；未把action+point总梯度冒充辅助梯度。
- 4个CPU进程以小型混合dtype模型（非完整π0.5）验证全局归约及DDP模型/Adam/RNG保存恢复，全部rank逐值一致。**不代表4张实体GPU的NCCL或速度已实测**。

| 单H100容量/计算探针 | 实测 |
|---|---|
| DDP、原生Adam、CUDA FP32 EMA、B32 | 峰值分配57.92GB，峰值reserved 63.77GB |
| B32预热后前向反向＋optimizer＋EMA | 1.978秒/次 |
| B4相同工作量：action-only前向反向 | 0.482秒 |
| B4加Pri4R前向反向 | 0.528秒，约增加9.6% |
| world=1完整保存＋校验 | 126.7–132.6秒；checkpoint目录约39.6GB，另有模型导出 |

容量探针重复一个真实样本，采用预热后的两次测量，包含CUDA EMA；不含数据加载、保存和多GPU通信。正式每1000步保存的摊销应结合目标集群实际step速度计算，不能把每步保存的短测速度当正常训练速度。实际四卡首批仍需核实world=4、每rank B32、global128、有限loss和完整保存；硬件拓扑/存储性能不由单卡探针保证。

源码manifest：`ba16e00d3a7840a439fdcf554a89e75d1d215b6d94610a9a92a8132f25695013`。旧ZeRO验收保留在[历史记录](../provenance/PRI4R_PAPER_VALIDATION_LEGACY649dd36.json)。本次只完成工程复现准备，没有全量GT、正式训练或成功率提升结论。
