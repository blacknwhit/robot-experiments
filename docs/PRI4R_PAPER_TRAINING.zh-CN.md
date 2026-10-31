# PI0.5 + Pri4R：RoboCasa365训练准备

## 状态与范围

本入口是按[Pri4R v2](https://arxiv.org/html/2603.01549v2)公开描述实现的独立版本，variant为`pri4r_paper`。截至2026-10-31核查时作者尚未公开训练代码，所以不能声称全部未公开细节与作者逐字节一致。论文实验为24原子任务/Human-50；本入口使用已有RoboCasa365的300任务训练数据，是方法迁移，不是论文分数复现。

**2026-10-31已通过真实单H100训练、完整保存、新进程恢复及无3D输入推理验收。** 本次只准备代码及有界验证，没有执行全量GT或正式训练。四卡配置和四进程归约检查通过；真实四卡NCCL及显存容量没有实测。下文全量命令供之后远端明确启动时使用。

## 固定方法与配方

- 原始PI0.5 base全量微调，不加载旧70k模型，不使用Query-FIXED、Memory或未来图像teacher。
- 独立可学习query读取VLM末层图文token；当前点经PointMLP，与时间特征经FusionMLP产生相邻3D位移。几何梯度回传VLM；动作路径不读取GT。推理时辅助分支不执行。
- 整条示范首帧从机器人周围裁剪的mesh表面采1024点，保持面片/重心坐标身份，逐帧恢复模拟器记录状态取得轨迹。机器人和场景统一按表面积采样；不采用旧版512机械臂/512接触对象配额，不依据未来接触或运动筛选点。
- `loss = 原flow matching loss + 1.0 * 相邻位移L1`；没有累计位移loss、位置loss、辅助loss ramp、梯度自适应配权或G8组采样。
- global batch128、30000优化步、AdamW、峰值LR5e-5、warmup10000、cosine、策略参数EMA0.999。默认4GPU×B32×acc1；恢复时同拓扑，不自动迁移rank数量。
- 完整状态保存raw模型、Adam、scheduler、RNG、数据进度、FP32 EMA；`steps_N_model.safetensors`是EMA评测权重，`steps_N_raw_model.safetensors`是raw对照。不要把raw或模型导出当成完整续训状态。

论文没有给出的细节明确写在`configs/pri4r_paper.yaml:experiment.local_conventions`：2层8头d2048辅助读取器、GELU两MLP宽度、2米cube半边长、窗口起点机器人底座坐标、米制、有效XYZ均值L1、EOF屏蔽、seed42、H50、cosine末值及AdamW细项。它们是透明的实现约定，不是作者确认参数。数据集采样沿用RoboCasa365官方300任务/100示范和长度^0.4任务权重。

## 1. 资源与环境

在已准备PI0.5/FIXED资源的集群复用原始数据、base及归一化统计，不重复下载。设定资源根：

```bash
export ROBOT_EXPERIMENTS_HOME=/absolute/path/to/pri4r_resources
export PRI4R_ASSET_ROOT="$ROBOT_EXPERIMENTS_HOME"
export EXPERIMENT_ROOT=/absolute/path/to/pri4r_experiment
export PY="$ROBOT_EXPERIMENTS_HOME/envs/policy/bin/python"
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"
```

资源目录需要：`data/robocasa365/{robocasa365-pretrain-atomic,robocasa365-pretrain-composite}`、`models/pi05_base_pytorch/model.safetensors`、`models/robocasa_norm/norm_stats.json`、`cache/openpi`。缺少资源时按仓库`HANDOFF.zh-CN.md`安装/下载，`scripts/download.py --component datasets/base/small`和`convert_base.py`仍可复用。新目录可以链接这些只读资产，训练输出使用独立`EXPERIMENT_ROOT`。

GT的`PRI4R_ASSET_ROOT`同时是文件边界：需要包含`playground/Code/{robocasa365,robosuite}`的真实资产、源示范文件和GT输出，解析符号链接后也必须在其中。新装环境默认等于资源根；复用旧资产时，把新资源目录放在旧资产根下面，并将`PRI4R_ASSET_ROOT`设为旧资产根。不要把资产链接到这个边界以外。

依赖安装与隔离overlay：

```bash
bash scripts/setup_policy.sh
"$PY" scripts/runtime_overlay.py --variant pri4r_paper
```

若已有兼容环境，仅执行第二条并核对版本即可。GT额外需要对应RoboCasa/robosuite的mesh assets；`scripts/setup_sim.sh`及原交接文档说明其获取方式。容器在创建时设置`NVIDIA_DRIVER_CAPABILITIES=all`；GPU短训需要NCCL可用。

## 2. 先做少量GT和短训

以下`data_scope.py`仅建立训练元数据清单，不生成全量GT。`prepare_pri4r_gt.py`默认只选4条示范；没有`--full`时拒绝超过8条。

```bash
"$PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage prepare
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage generate --workers 2
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage verify

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 1 --smoke --stop 2 \
  --out resolved/pri4r_smoke2.yaml
"$PY" scripts/import_preflight.py --config resolved/pri4r_smoke2.yaml
"$PY" scripts/train.py --config resolved/pri4r_smoke2.yaml

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 1 --smoke --stop 3 \
  --out resolved/pri4r_smoke3.yaml
"$PY" scripts/train.py --config resolved/pri4r_smoke3.yaml --resume
"$PY" scripts/point_inference_probe.py --config resolved/pri4r_smoke3.yaml \
  --checkpoint "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/checkpoints/steps_3_model.safetensors" \
  --sample "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/engineering_sample.pt" \
  --output "$EXPERIMENT_ROOT/pri4r_inference_check.json"
```

短训使用真实RGB/action/GT，限制到清单里的示范，不会从未准备GT的其他episode随机补样。损坏/缺失GT直接报错；正式模式拒绝绑定短测清单。验收检查每层Embedding、点头、共享VLM有限非零梯度和FP32更新，以及各rank恢复和EMA clock。推理检查完整EMA加载、训练/缓存路径一致及不读取3D点的有限动作。

单卡短训自动使用B4、梯度累积32、global128；torch AdamW状态与FP32 EMA放在CPU，模型和前向反向仍在GPU。该设置只用于有界验收，要求足够CPU内存；不能将单卡CPU offload短测速度作为正式四卡训练速度。`--gpus 1 --formal`会直接拒绝。

四卡正式配置仍是每卡B32、acc1、global128，Adam与EMA均在CUDA。四进程CPU/Gloo检查可单独执行：

```bash
"$PY" -m unittest discover -s tests -p test_pri4r_reduction.py -v
```

它检查不等有效点数及累积下的分布式梯度与全batch串行参考等价，使用本机TCP rendezvous以避免共享盘文件锁问题。它不代表四张GPU的NCCL、显存容量或完整ZeRO2实机验收。**资源不足不要自行减global batch或改变loss/EMA。**

## 3. 以后在远端全量准备GT

这一步耗时、占空间，只在明确要正式训练时执行。使用全新的资源目录，不覆盖上述短测目录，也不覆盖旧armobj标签。例如在上述资产根下建立`pri4r_full`作为新的`ROBOT_EXPERIMENTS_HOME`，链接同一资产根内的RGB/base/cache，保持`PRI4R_ASSET_ROOT`不变；源文件及GT输出都应位于该资产根内。`PY`可继续指向已安装的环境，切换资源根后同步设置`OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"`。不要把短测的`input_manifest.json`直接当全量清单。

```bash
"$PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage prepare --full
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage generate --full \
  --workers 6 --num-shards 8 --shard-index 0
# 其余CPU作业分别使用shard-index 1..7，保持同一份配置/manifest。
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --asset-root "$PRI4R_ASSET_ROOT" --stage verify --full
```

完整GT仍约350GB量级，另留原始源和中间缓存空间。完整校验要求30000条/27815381帧及每条源身份、配置、数组SHA；校验产物是`data/gt_pri4r_paper/PORTABLE_GT_ACCEPTED.json`。几何采样由`rigid_surface_replay_v1`执行，不能用旧armobj的完成marker代替。

## 4. 以后正式训练及续训

```bash
"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_paper_seed42 --gpus 4 --formal --stop 30000 \
  --out resolved/pri4r_formal.yaml
"$PY" scripts/train_or_resume.py --config resolved/pri4r_formal.yaml
```

第一次从原base初始化；以后只选择最新完整兼容状态恢复，不从工程权重启动，不在缺state时静默重来。保持同一固定checkout和resolved配置；新checkout源SHA不同，应保留旧release续训。SCO提交默认开启原生`autoRetrain=true, autoRetrainIntervalMinutes=10`；其他集群用其任务调度重启同一条命令，不建立第二个竞争writer。

EMA按每次真实optimizer update更新，完整checkpoint不可缺EMA或更新数不一致。模型评测复用现有pipeline，指定`--variant pri4r_paper`及对应config、`steps_N_model.safetensors`；统一使用同一2500回合种子协议，旧历史分数不能冒充逐场景配对结果。

## 验证证据

完整记录见[PRI4R_PAPER_VALIDATION.json](../provenance/PRI4R_PAPER_VALIDATION.json)。受测源manifest SHA：`4abd3918d42b5ed0b46502701faacb453959dd14ac9f94da373b833e8ca69616`。

- 6项配置/范围契约、3项梯度/损失/EMA数学检查、1项四进程Gloo归约，以及4项旧入口回归通过。
- 真实GT：4条示范、1126帧、每条1024点，源对应及所有数组SHA校验通过。
- 实机：1×H10080GB、B4/acc32/global128，原始base全量微调；仅3次优化更新。实际AdamW CPU offload、FP32 EMA在CPU，训练阶段Torch峰值分配23.54GB；此数不代表完整保存期间的总显存，也不能推导四卡B32容量。
- step1/2/3完整保存通过。新进程从step2恢复，11156733755个FP32主权重/Adam值逐位一致，BF16、RNG、padding及EMA均通过审计，数据进度64→96 microbatch、EMA更新数2→3。
- EMA导出严格加载861项、0项丢弃；两种噪声的训练前向/缓存推理差值均0。移除所有point输入并禁止辅助分支执行后，动作`[1,50,12]`有限，同seed两次完全一致。

| update | action loss | adjacent 3D L1 | total loss | gradient norm |
|---|---|---|---|---|
| 1 | 0.489077 | 0.129076 | 0.618153 | 0.794764 |
| 2 | 0.510510 | 0.128540 | 0.639050 | 0.817635 |
| 3（恢复后） | 0.532478 | 0.129327 | 0.661806 | 0.831959 |

每步共享VLM、两层Embedding及点头的12项审计参数均有有限非零梯度与FP32主权重更新。极小warmup LR下部分BF16参数差值日志为0，不能以此替代FP32审计。日志中的`FORESIGHT_4GPU_RESTORE_AUDIT_OK`是继承的历史名称，实际world以记录中的1为准。

这些证据验证训练和续训接线，不证明3步收敛、正式训练稳定性或性能提升；没有闭环benchmark结果。正式四卡仍需在目标机器观察首批更新与保存，不能把本次单卡验证称为四卡实机验证。
