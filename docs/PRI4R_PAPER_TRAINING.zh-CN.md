# PI0.5 + Pri4R：RoboCasa365训练准备

## 状态与范围

本入口是按[Pri4R v2](https://arxiv.org/html/2603.01549v2)公开描述实现的独立版本，variant为`pri4r_paper`。作者尚未公开训练代码，所以不能声称全部未公开细节与作者逐字节一致。论文实验为24原子任务/Human-50；本入口使用已有RoboCasa365的300任务训练数据，是方法迁移，不是论文分数复现。

**候选尚待真实GPU短训验收；本页会在验收后附对应证据。不要把CPU测试通过当作训练已通过。** 本次只准备代码及有界验证，不在SCO执行全量GT或正式训练。下文全量命令供之后远端明确启动时使用。

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
export EXPERIMENT_ROOT=/absolute/path/to/pri4r_experiment
export PY="$ROBOT_EXPERIMENTS_HOME/envs/policy/bin/python"
```

资源目录需要：`data/robocasa365/{robocasa365-pretrain-atomic,robocasa365-pretrain-composite}`、`models/pi05_base_pytorch/model.safetensors`、`models/robocasa_norm/norm_stats.json`、`cache/openpi`。缺少资源时按仓库`HANDOFF.zh-CN.md`安装/下载，`scripts/download.py --component datasets/base/small`和`convert_base.py`仍可复用。新目录可以链接这些只读资产，训练输出使用独立`EXPERIMENT_ROOT`。

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
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage prepare
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage generate --workers 2
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage verify

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 4 --smoke --stop 2 \
  --out resolved/pri4r_smoke2.yaml
"$PY" scripts/import_preflight.py --config resolved/pri4r_smoke2.yaml
"$PY" scripts/train.py --config resolved/pri4r_smoke2.yaml

"$PY" scripts/configure.py --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME" \
  --experiment-root "$EXPERIMENT_ROOT" --run-id pri4r_smoke --gpus 4 --smoke --stop 3 \
  --out resolved/pri4r_smoke3.yaml
"$PY" scripts/train.py --config resolved/pri4r_smoke3.yaml --resume
"$PY" scripts/point_inference_probe.py --config resolved/pri4r_smoke3.yaml \
  --checkpoint "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/checkpoints/steps_3_model.safetensors" \
  --sample "$EXPERIMENT_ROOT/smoke_runs/pri4r_smoke/engineering_sample.pt" \
  --output "$EXPERIMENT_ROOT/pri4r_inference_check.json"
```

短训使用真实RGB/action/GT，限制到清单里的示范，不会从未准备GT的其他episode随机补样。损坏/缺失GT直接报错；正式模式拒绝绑定短测清单。验收检查每层Embedding、点头、共享VLM有限非零梯度和FP32更新，以及各rank恢复和EMA clock。推理检查完整EMA加载、训练/缓存路径一致及不读取3D点的有限动作。

**资源不足不要自行减global batch或改变loss/EMA。** 目前支持的GPU容量与实际短测证据以后附在此页；4卡B32的命令本身不是容量已通过证明。

## 3. 以后在远端全量准备GT

这一步耗时、占空间，只在明确要正式训练时执行。使用全新的资源根或全量GT目录，不覆盖上述短测目录，也不覆盖旧armobj标签。新资源根仍可链接同一份RGB/base/assets。

```bash
"$PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage prepare --full
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage generate --full \
  --workers 6 --num-shards 8 --shard-index 0
# 其余CPU作业分别使用shard-index 1..7，保持同一份配置/manifest。
"$PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" --stage verify --full
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

当前：5项CPU配置/范围契约、3项CPU梯度/损失/EMA数学检查通过；真实GT与GPU门禁进行中。最终验证范围、GPU/代码SHA、恢复及推理结果会单独封存；本次不产生正式benchmark分数。
