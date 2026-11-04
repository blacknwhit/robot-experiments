# 给远端Astra：C3动作条件3D辅助训练，8×H100，70k

本页是C3的独立执行入口。目标：复用已有RoboCasa365与旧arm/object GT，从原始π0.5底座完成70000次优化更新，完成完整state及策略权重保存。所有代码已在`experiments/c3_action/`提供；不要在远端另行重写模型或将`pri4r_paper`入口当作C3。

## 1. 固定实验定义

| 项目 | 本版设置 |
|---|---|
| 模型 | 原π0.5动作路径＋独立Embedding读出＋当前点MLP＋因果动作GRU＋融合MLP |
| 动作条件 | 同一示范同一窗口的干净GT动作，π0.5预处理后取50×12，经12→64/GELU/GRU64/64→256；只进入辅助头 |
| 时间/点融合 | 50个2048维时间槽，2层self/cross/FFN；点MLP 3→128→2048；两路投影256后加动作投影，GELU→3 |
| GT | 旧`gt_armobj_full_v1`，1024点，单位米；以窗口起始机器人基座固定坐标系预测50段相邻位移 |
| 辅助损失 | `4 × (相邻有效XYZ平均L1 + mean(累计10/10、累计25/25、累计50/50的L1))`；累计mask要求整段有效 |
| 总损失 | action FM＋`0.07516936569041761 × min(1, update/100) × 辅助损失` |
| 数据 | 300任务×100示范；现有baseline按任务权重抽样、G8 ordered-random，global64 |
| GPU/batch | 单机8×H10080GB，每卡B8，acc1；不外套第二层torchrun |
| 更新/日程 | **训练到70k停止；保留原C3的100k cosine日程、1000步warmup**，不将cosine缩成70k |
| LR | 主模型2.5e-5、辅助头1e-4，沿同一倍率衰减；主模型日程末值2.5e-6 |
| 优化器 | native torch AdamW，beta(.9,.95)、eps1e-8、WD1e-10、foreach=False；主/辅助分别clip1 |
| 精度 | 全部可训练参数及Adam为FP32，前向BF16 autocast；TF32关闭、保留activation checkpointing |
| EMA | **不使用EMA**，与本次C3初筛一致；评测导出在线训练权重 |
| 保存 | 完整state在update1、每5000及70000；每次同时导出无辅助头策略 |

这是原C3方法的全量扩展，不是2000步初筛的逐样本重复：初筛只有8任务×50训练示范、固定邻近窗口缓存和每更新8任务均衡抽样；本版改用全量baseline任务权重/G8采样。辅助模块、损失、权重、优化器、LR日程保留。起始frame0可能跨两个物理控制间隔，本版保留该action样本但将整段辅助mask置false；尾端不足50步也按有效mask处理，不补假轨迹。不要混用论文复现的3m均匀采点GT、世界坐标loss、EMA或paper学习率。

辅助系数来自初始8样本共享VLM梯度校准，不能声称全量训练中始终是15%梯度比。C0/C1同入口可用于另行授权的对照，默认只跑C3。若需要严格隔离辅助任务收益，可另行用相同全量配方的`--arm C0`；历史baseline的精度/采样/随机流未必完全相同，不默认称其为只差一个辅助分支的对照。

## 2. 代码与已有资源

使用本次发布commit的独立checkout，训练开始后冻结。保留旧结果，不在运行中pull或覆盖配置。

```bash
set -euo pipefail
git clone https://github.com/blacknwhit/robot-experiments.git robot-experiments-c3
cd robot-experiments-c3
export CODE_DIR="$PWD"
export ROBOT_EXPERIMENTS_HOME=/absolute/path/to/existing-resources
export GT_ROOT="$ROBOT_EXPERIMENTS_HOME/data/gt_armobj_full_v1"
export EXPERIMENT_ROOT=/absolute/path/to/new-c3-experiment
export POLICY_PY="$ROBOT_EXPERIMENTS_HOME/envs/policy/bin/python"
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"
mkdir -p "$EXPERIMENT_ROOT/resolved" "$EXPERIMENT_ROOT/logs"
git rev-parse HEAD > "$EXPERIMENT_ROOT/code-commit.txt"
```

资源根必须已有：

- `data/robocasa365/robocasa365-pretrain-atomic`和`robocasa365-pretrain-composite`；固定数据revision `522e4ffa3bb9d9729f79b241c65fc9e589f2659c`。
- `models/pi05_base_pytorch/model.safetensors`，必须原始base，不是已训练70k权重。
- `models/robocasa_norm/norm_stats.json`、`cache/openpi/big_vision/paligemma_tokenizer.model`、Arrow cache。
- `GT_ROOT/sampling_config.json`及30000示范的完整GT子目录，保留`complete.json`和所有点身份/位姿/时钟数组。

资源缺失按[主交接文档](../HANDOFF.zh-CN.md)下载数据、base并准备缓存；已有资源直接复用。已有兼容policy训练环境不必重装；确需新建时执行`bash scripts/setup_policy.sh`。policy使用Python3.11、Torch2.7.1+cu128、Transformers4.53.2、NCCL2.26.5，保留原环境freeze。

```bash
"$POLICY_PY" scripts/runtime_overlay.py --variant embedding3d_v2
"$POLICY_PY" experiments/c3_action/test_cpu.py
```

本训练使用native DDP而非DeepSpeed/FSDP，不进行跨world的optimizer重分片。若8卡容器不足以容纳B8，保留日志并报告；不要在中途偷偷改变batch、精度或模型。

## 3. 直接复用并核验旧GT

本节在CPU节点完成即可，读取约350GB已有GT会产生较多I/O；不要先占用8张H100等待GT校验。第4、5节才进入GPU训练节点。

```bash
"$POLICY_PY" experiments/c3_action/prepare.py verify-gt --gt-root "$GT_ROOT" \
  2>&1 | tee "$EXPERIMENT_ROOT/logs/gt-verify.log"
```

这是读取已有数组并校验SHA、来源、采点配置、时钟、30000示范/27815381帧，**不会重跑仿真或重新采点**。只新增`C3_GT_ACCEPTED.json`凭证，不改已有GT。成功后本次资源可复用此凭证；模型换结构不需要重新生成轨迹。不要删掉检查来强行使用`gt_pri4r_paper`或缺少metadata的数组。

所有训练读取按source episode与frame_index对应观察、50步GT动作和51帧点位置，不按聚合数据中的episode编号猜映射。`GT_ROOT`可在已有资源根之外，通过参数给出真实绝对路径即可。

## 4. 新训练入口先做一次保存/恢复检查

本版尚未宣称8张实体H100全模型验收。新增全量训练入口首次部署需要以下三更新检查；这些是独立smoke输出，正式训练不会接着它们训。本交付只处理训练，不安排benchmark评测。

```bash
set -euo pipefail
"$POLICY_PY" experiments/c3_action/prepare.py configure \
  --home "$ROBOT_EXPERIMENTS_HOME" --gt-root "$GT_ROOT" \
  --run-root "$EXPERIMENT_ROOT/smoke_runs" --run-id c3_h8_check \
  --gpus 8 --mode smoke --schedule-steps 100000 \
  --out "$EXPERIMENT_ROOT/resolved/c3-smoke.json"

"$POLICY_PY" experiments/c3_action/run.py --config "$EXPERIMENT_ROOT/resolved/c3-smoke.json" --stop 2 \
  2>&1 | tee "$EXPERIMENT_ROOT/logs/smoke2.log"
"$POLICY_PY" experiments/c3_action/run.py --config "$EXPERIMENT_ROOT/resolved/c3-smoke.json" --stop 3 \
  2>&1 | tee "$EXPERIMENT_ROOT/logs/resume3.log"
```

检查8个`ATTEMPT_RANK*.json`均为`start_step=2`、`model_optimizer_rng_exact=true`、world8；每rank新step3 loss/grad均有限；step3完整保存及策略导出存在。保存时还比较所有rank的完整模型/Adam签名；恢复逐值校验模型/Adam并恢复每rank RNG。无梯度的参数没有Adam state是合法状态，不填造step或零动量。

如只有单H100可用于工程检查，独立run配置`--gpus 1 --mode smoke`，使用B8/acc8/global64；这不能替代首次8rank/NCCL/显存检查。两种smoke不得同run切换卡数。

## 5. 正式8卡训练至70k

```bash
"$POLICY_PY" experiments/c3_action/prepare.py configure \
  --home "$ROBOT_EXPERIMENTS_HOME" --gt-root "$GT_ROOT" \
  --run-root "$EXPERIMENT_ROOT/runs" --run-id c3_h8_b8_s42_70k \
  --gpus 8 --mode formal --schedule-steps 100000 \
  --out "$EXPERIMENT_ROOT/resolved/c3-70k.json"

"$POLICY_PY" experiments/c3_action/run.py --config "$EXPERIMENT_ROOT/resolved/c3-70k.json" --dry-run
"$POLICY_PY" experiments/c3_action/run.py --config "$EXPERIMENT_ROOT/resolved/c3-70k.json" \
  2>&1 | tee "$EXPERIMENT_ROOT/logs/train.log"
```

仅启动一次`run.py`，它内部启动8个rank，B8×8×acc1=64。全量微调从原始base开始，包含辅助头在内的FP32参数/Adam和梯度常驻GPU；保留checkpointing。建议提供充足主存与本地/共享SSD：rank0完整state为数十GB，每rank保存独立RNG；默认每rank4个数据worker，全机32个。每rank metrics独立记录，rank0打印的是rank0局部batch loss，不伪称全局均值。

中断后重跑完全相同的`run.py --config ...`命令。它自动选最新完整且recipe/world兼容的state，恢复模型、Adam、LR步数、每rank RNG，以及确定性global sample index。数据worker预取不会提前消耗已提交的数据进度。临时/不完整保存被跳过；启动claim后没有完整state则报错，绝不从base静默重跑。保持代码、resolved配置、资源指纹、8卡拓扑和run-id不变。

SCO提交时按项目要求使用单一原生10分钟自动恢复；其它集群使用一个调度恢复者即可。文件锁防双writer，不另写多个竞争重启循环。本发布不包含远端具体调度平台的提交API。

完成以以下文件为准，不能只看平台结束：

```text
runs/c3_h8_b8_s42_70k/FINISHED.json                 # step=70000
runs/c3_h8_b8_s42_70k/checkpoints/step_070000/      # training.pt + 8个RNG + COMPLETE
runs/c3_h8_b8_s42_70k/exports/step_070000/          # 无辅助头model.safetensors + COMPLETE
```

70000是训练停止点，学习率仍位于原100k日程中；不要以“未衰减到最低”误判训练没结束。默认不删除旧检查点：按初筛state约45GB、导出约8GB估算，保留15次保存约需0.8TB，另加数据、GT和缓存；这是容量估计，以实际文件为准。

## 6. 训练报告与验收边界

向用户报告：固定commit、GT凭证、resolved配置、8rank首次恢复验收、实际70000完整state与无辅助头导出、训练损失曲线、运行时间与错误日志。不要把smoke、只读检查或平台RUNNING写成70k完成。

原C3模型在SCO上已有2000步及160回合闭环证据；本发布新增的是全量portable数据调度/native DDP入口。新增CPU测试覆盖GRU因果性、三路融合与原实现数值一致、梯度路径、mask/累计损失、全状态恢复、缺失Adam条目和跨rank采样索引；具体本地执行结果见`provenance/C3_PORTABLE_VALIDATION.json`。尚无本版8×H100全模型实跑或70k性能承诺，因此第4节是新集群首次部署的实际验收步骤。
