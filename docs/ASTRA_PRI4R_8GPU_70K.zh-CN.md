# 远端直接执行：Pri4R × PI0.5，8卡训练70k与评测

本页是2026-11-01用户指定的正式训练交接。使用公开仓库版本标签 **`pri4r-h8-70k-20261101`**，配置profile **`pri4r_8gpu70k_v1`**。远端按以下命令准备资源、启动训练、恢复和评测，不需再自行修改模型或评测代码。本地没有替远端生成全量GT或启动正式训练。

## 1. 最终配置与适用范围

| 项目 | 本次正式配置 | 依据 |
|---|---|---|
| 训练硬件 | 单机8×H100 80GB；原生DDP，8个rank | 本次8卡交接；尚未实体8卡联跑验收 |
| batch | **每卡16 × 8卡 × 梯度累积1 = global128** | Pri4R全局batch；用户要求关闭梯度累积 |
| 优化步数 | **max=70000、stop=70000、cosine总区间=70000** | 用户覆盖论文30k预算；不是只改停训步数 |
| 学习率 | peak `5e-5`，warmup `10000`，cosine末值 `2.5e-6` | Pri4R峰值/预热；末值沿用官方π0.5默认 |
| 优化器 | AdamW，betas `.9/.95`、eps `1e-8`、WD `1e-10`、clip `1` | 原π0.5训练配置；不改参数组策略 |
| EMA | `.999`，每次update更新FP32 shadow，正式放GPU | Pri4R；评测使用EMA导出 |
| 初始化/数据 | 原始 `pi05_base`；RoboCasa365 300任务×100示范，共30000条、27815381帧 | 已训π0.5基线的资源与数据选择 |
| 数据采样 | IID frame、filter `100_demos`/seed0、任务权重alpha `.4`；训练seed42 | 本次母结构及基线数据路径；不引入G8 |
| 辅助任务 | 1024点、50段相邻位移、世界坐标米制、有效XYZ平均L1×1 | 当前Pri4R复现约定；不加累计/位置loss |
| 辅助结构 | 独立2层Embedding，2048维/8头；Point/Fusion MLP；不插入动作prefix | 已核对并短测的Pri4R实现约定 |
| 精度/显存 | 官方selective BF16/FP32，原生Adam；activation checkpointing和fusion checkpointing保持开启 | 保留已验证的实现；不是全模型BF16或ZeRO |
| 保存/日志 | update1额外完整保存，其后每5000步；日志每100步 | 保存间隔与已训π0.5基线对齐 |
| loader | 每rank 8 workers、prefetch4、pin memory；OpenCV视频读取 | 与已训π0.5基线对齐，共64个loader worker |

**“关梯度累积”是 `gradient_accumulation_steps=1`；不关闭 `gradient_checkpointing`。** 后者用重计算降低显存，不改变global batch。

Pri4R优先的配置与旧基线并非完全相同：旧基线global64、peak2.5e-5、warmup1k、100k cosine且停70k；本次global128、peak5e-5、warmup10k、70k cosine及EMA。70k对应本次896万、旧基线448万训练样本抽取，不能称为只增加辅助loss的等预算消融。作者未公开的2层Embedding/米制world/3m采样立方体/mean L1等细节采用用户已确认的约定；这也不是论文原数据集分数的复现。

## 2. 获取固定代码与复用环境

以下命令使用Linux Bash。先把路径改成目标集群的绝对路径；代码使用新的独立目录，不更新仍在运行的旧checkout。

```bash
set -euo pipefail
export CODE_DIR=/absolute/path/to/robot-experiments-pri4r-h8-70k
git clone --branch pri4r-h8-70k-20261101 --single-branch \
  https://github.com/blacknwhit/robot-experiments.git "$CODE_DIR"
cd "$CODE_DIR"
git rev-parse HEAD
```

优先复用先前已经能训练π0.5、运行本仓库评测的环境。`SHARED`指已有的资源根，包含下面列出的models/data/cache和真实仿真资产；不是任意父目录。若之前只有4条短测GT或旧时钟GT，必须使用新的Pri4R资源子目录，保留旧产物。

```bash
export SHARED=/absolute/path/to/existing-pi05-resources
export ROBOT_EXPERIMENTS_HOME="$SHARED/pri4r-h8-70k-resources"
export PRI4R_ASSET_ROOT="$SHARED"
export EXPERIMENT_ROOT="$SHARED/experiments/pri4r-h8-70k"
export POLICY_PY="$SHARED/envs/policy/bin/python"
export SIM_PY="$SHARED/envs/sim/bin/python"
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"

# 只建立新目录内的链接，不覆盖已有数据或权重。
link_once() {
  local src=$1 dst=$2
  [[ -e "$src" ]] || { echo "Missing resource: $src"; return 1; }
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" || -L "$dst" ]]; then
    [[ "$(readlink -f "$dst")" == "$(readlink -f "$src")" ]] || {
      echo "Existing target differs: $dst"; return 1;
    }
  else
    ln -s "$src" "$dst"
  fi
}
link_once "$SHARED/models" "$ROBOT_EXPERIMENTS_HOME/models"
link_once "$SHARED/data/robocasa365" "$ROBOT_EXPERIMENTS_HOME/data/robocasa365"
link_once "$SHARED/cache/openpi" "$ROBOT_EXPERIMENTS_HOME/cache/openpi"
link_once "$SHARED/cache/arrow" "$ROBOT_EXPERIMENTS_HOME/cache/arrow"
link_once "$SHARED/receipts/policy.freeze.txt" "$ROBOT_EXPERIMENTS_HOME/receipts/policy.freeze.txt"
link_once "$SHARED/receipts/sim.freeze.txt" "$ROBOT_EXPERIMENTS_HOME/receipts/sim.freeze.txt"
"$POLICY_PY" scripts/runtime_overlay.py --variant pri4r_paper
```

资源根的必要布局：

```text
SHARED/
  models/pi05_base_pytorch/model.safetensors
  models/robocasa_norm/norm_stats.json
  data/robocasa365/robocasa365-pretrain-atomic/
  data/robocasa365/robocasa365-pretrain-composite/
  cache/openpi/big_vision/paligemma_tokenizer.model
  cache/arrow/
  playground/Code/robocasa365/robocasa/models/assets/
  playground/Code/robosuite/robosuite/models/assets/
  receipts/{policy,sim}.freeze.txt
```

`PRI4R_ASSET_ROOT`需要同时包含解析链接后的源数据、GT输出和上述mesh路径；GT工具严格限制文件边界。若现有资源分散在其它位置，先整理到这个布局再继续，不能只改变量骗过校验。已有环境不要无条件重跑安装脚本：已验证组合为Python3.11、Torch2.7.1+cu128、Transformers4.53.2、Accelerate1.5.2、NCCL2.26.5；sim为MuJoCo3.3.1及固定robosuite/RoboCasa版本。训练不需要登录宿主GPU。

**如果是全新集群、没有可复用资源**，改走下面独立分支，跳过上面的链接命令：

```bash
export ROBOT_EXPERIMENTS_HOME=/absolute/path/to/new-pri4r-resources
export PRI4R_ASSET_ROOT="$ROBOT_EXPERIMENTS_HOME"
export EXPERIMENT_ROOT=/absolute/path/to/new-pri4r-experiment
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"
export POLICY_PY="$ROBOT_EXPERIMENTS_HOME/envs/policy/bin/python"
export SIM_PY="$ROBOT_EXPERIMENTS_HOME/envs/sim/bin/python"
command -v uv >/dev/null || python3 -m pip install uv
bash scripts/setup_policy.sh
bash scripts/setup_sim.sh
"$POLICY_PY" scripts/download.py --home "$ROBOT_EXPERIMENTS_HOME" --component small
"$POLICY_PY" scripts/download.py --home "$ROBOT_EXPERIMENTS_HOME" --component datasets
"$POLICY_PY" scripts/download.py --home "$ROBOT_EXPERIMENTS_HOME" --component base
"$POLICY_PY" scripts/convert_base.py --home "$ROBOT_EXPERIMENTS_HOME"
"$POLICY_PY" scripts/prepare_arrow.py --home "$ROBOT_EXPERIMENTS_HOME"
```

安装/下载/GT生成先在CPU准备任务中完成，不占用8卡训练作业等待。`convert_base.py`只对尚不存在的转换输出执行一次。norm取自公开RoboCasa资产，但初始化权重必须是原始pi05_base，不能用75k policy、历史70k或短测工程权重替代。

存储需另预留全量GT、下载的源states/XML、转换cache和完整checkpoint。仅float32 xyz轨迹主数组约342GB（27815381×1024×3×4字节），另有mask、时间轴及metadata；本配置不自动删除旧checkpoint，按首次完整保存的实际大小为step1＋14个5k保存点和导出留足空间，不把仅推理权重大小当作完整state大小。

## 3. 全量GT：执行完整流程，不只生成一个分片

```bash
"$POLICY_PY" scripts/data_scope.py --home "$ROBOT_EXPERIMENTS_HOME"
"$POLICY_PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" \
  --asset-root "$PRI4R_ASSET_ROOT" --stage prepare --full
"$POLICY_PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" \
  --asset-root "$PRI4R_ASSET_ROOT" --stage generate --full \
  --workers 8 --num-shards 1 --shard-index 0
"$POLICY_PY" scripts/prepare_pri4r_gt.py --home "$ROBOT_EXPERIMENTS_HOME" \
  --asset-root "$PRI4R_ASSET_ROOT" --stage verify --full
```

这里是一个入口加8个CPU worker，`num-shards=1`涵盖全部30000条示范，不是只处理1/8数据。prepare整理当前数据行和manifest；generate中的worker下载并核对源states/XML/meta/时间轴，因此generate阶段也需网络或已经缓存的源文件。verify应产生 `data/gt_pri4r_paper/PORTABLE_GT_ACCEPTED.json`：`bounded=false`、30000 episodes、27815381 frames、`physical_window_alignment_v1`、全数组hash通过。不能把4条GT凭证改成全量。

真实simulator clock中跨2dt的首段只屏蔽对应辅助窗口，保留π0.5原始动作/RGB行；正常窗口不变。旧GT若没有这个策略，放在旧目录保留，不能直接复用。已有**同策略完整GT**可保留其资源home并重跑verify，省略prepare/generate；这不包括旧arm/object GT。

## 4. 生成并检查正式70k配置

```bash
cd "$CODE_DIR"
export RUN_ID=pri4r_h8_70k_s42
export CONFIG="$CODE_DIR/resolved/pri4r-h8-70k-s42.yaml"
mkdir -p "$CODE_DIR/resolved" "$EXPERIMENT_ROOT"
"$POLICY_PY" scripts/configure.py --variant pri4r_paper \
  --training-profile pri4r_8gpu70k_v1 \
  --home "$ROBOT_EXPERIMENTS_HOME" --experiment-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" --gpus 8 --formal --stop 70000 --out "$CONFIG"
"$POLICY_PY" scripts/pri4r_preflight.py --config "$CONFIG"
"$POLICY_PY" scripts/import_preflight.py --config "$CONFIG"
"$POLICY_PY" scripts/train_or_resume.py --config "$CONFIG" --dry-run
```

configure只运行一次，拒绝覆盖已有resolved YAML。preflight检查源码、完整GT、scope、base/norm，不分配模型；它不代替实体8卡测试。dry-run必须打印 `world=8, micro_batch=16, accumulation=1, global_batch=128, max_train_steps=70000, stop_after_steps=70000, cosine_decay_steps=70000, save_interval=5000, ema_device=cuda`。

不要省略 `--training-profile`：旧论文30k模板保留作为历史配置，不能只手工把stop改70k。不要预先创建 `$EXPERIMENT_ROOT/runs/$RUN_ID`，否则入口会按已有实验执行恢复；创建其父目录即可。

## 5. 8卡训练与自动恢复

在单机8张H100均已分配的任务中，恢复上面的环境变量并执行：

```bash
cd "$CODE_DIR"
export OPENPI_DATA_HOME="$ROBOT_EXPERIMENTS_HOME/cache/openpi"
"$POLICY_PY" scripts/train_or_resume.py --config "$CONFIG"
```

只启动**一个**上述命令；内部自动torchrun启动8rank。不能再外套8进程torchrun，也不能每卡分别启动8份训练入口。保存任务的stdout/stderr和完整run目录。机器需提供64个loader worker与8个训练rank所需的CPU/I/O资源；首段实际训练应核实8rank、B16/acc1/global128、有限action/point loss与参数更新、update1完整保存。原生DDP通信性能、正式全量数据吞吐仍以目标机器日志为准，不保证8卡线性加速。

重启时使用**同一份代码、resolved YAML、run-id和完全相同的命令**。入口自动选最新完整兼容checkpoint，恢复模型、Adam、scheduler、8rank RNG、数据进度及EMA；跳过不完整保存。如果run已存在但没有完整state，直接报错，不静默重新从base开始。不要接续旧30k配方、四卡或短测权重，也不要在中途更换卡数/精度/源码。writer锁防止同一个run双写。

若在SCO提交，启用并核验原生 `autoRetrain=true, autoRetrainIntervalMinutes=10`；以monitor的currentJob为准，不另写竞争重试循环。其它集群使用其已有单一恢复机制，避免两个调度者同时启动本run。第一次启动失败也先看真实异常，不盲目改batch或丢弃run。

checkpoint为step1、5000、10000、…、70000。满70k后退出；检查：

```bash
"$POLICY_PY" scripts/pri4r_preflight.py --config "$CONFIG" --completed
export CHECKPOINT="$EXPERIMENT_ROOT/runs/$RUN_ID/checkpoints/steps_70000_model.safetensors"
```

`--completed`要求实际完整兼容checkpoint-70000，验证模型、优化器、各rank状态/时钟、EMA及推理导出；最新只有65000不算完成。`steps_70000_model.safetensors`是EMA推理权重；`steps_70000_raw_model.safetensors`是在线训练权重，不替换默认评测权重。

## 6. 8卡正式评测：固定2500种子、多任务流水线

训练结束后可复用8卡节点运行评测，释放训练进程后再启动，避免同卡竞争。评测是每卡一份policy服务、独立CPU仿真worker与绑定该卡的共享渲染服务，不是DDP训练。需要容器在创建时设置 **`NVIDIA_DRIVER_CAPABILITIES=all`** 并提供EGL；仅在已启动容器内export不能补回缺失的驱动挂载。

下面以CUDA与EGL编号在任务容器内均对应0–7为例。若集群重映射设备，先按UUID/PCI核实并修改 `--egl-device-ids` 对应关系，不把每个policy内的逻辑cuda:0都当成EGL0。

```bash
cd "$CODE_DIR"
export EVAL_OUT="$EXPERIMENT_ROOT/eval/pri4r-h8-70k-fixed-seeds-v1"
EVAL_ARGS=(
  --variant pri4r_paper --home "$ROBOT_EXPERIMENTS_HOME"
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --sim-python "$SIM_PY"
  --output "$EVAL_OUT" --gpu-ids 0,1,2,3,4,5,6,7
  --egl-device-ids 0,1,2,3,4,5,6,7 --port-base 24090
  --n-envs 16 --tasks-per-gpu 50 --simulator-cpu-threads 1
  --rgb-schedule final --initial-rgb deferred --timeout 172800
)
"$POLICY_PY" scripts/evaluate_parallel.py "${EVAL_ARGS[@]}" --all-tasks
```

- 完整50任务×每任务50回合=2500；使用仓库固定episode种子表及逐请求模型噪声，改并发数不换seed。种子表SHA256为 `b1c5243b89f4d6283f01cb950bb93911269e8b9068f334062a869d604dca1c9b`。
- `n-envs=16`是**每张GPU合计16个CPU仿真worker**，8卡最多128个；`tasks-per-gpu=50`表示任务池上限，不是50份模型或每任务16环境。动态均衡分配实际剩余任务，减少某个慢任务让卡空等。
- H50、每5步重新规划、10步去噪、task清单原horizon（不乘1.5）；原混合精度、TF32关闭。完整EMA加载后去除辅助模块，模型不接收点轨迹GT。
- `final/deferred`省掉控制中间步和暂不读取时的RGB，不改变物理step或实际供模型读取的观测时刻。CUDA Graph/pipeline使用现有验证路径，保留自动动作数值检查及黑图/异常检查；不要求生成场景计划，不作旧参考初始图像逐像素hash比较。
- 默认直接正式评测，不先重复50任务smoke。这个配置以既有单H100高吞吐证据为起点；当前Pri4R8卡/全量GT尚未实跑，不能承诺整机或3小时窗口固定84%–85%利用率。CPU不足/内存紧张时先用每卡8环境，充分供给CPU后再上16；不同配置使用新的输出目录。

整个评测完成会自动生成 `$EVAL_OUT/FINAL_ACCEPTED.json`，要求50任务、2500个唯一回合、原始成功布尔值及证据完整。可重新汇总而不再推理：

```bash
"$POLICY_PY" scripts/evaluate_parallel.py "${EVAL_ARGS[@]}" --summarize
```

中断后在相同代码/模型/config/参数下重跑原 `--all-tasks` 命令，已完整提交且身份一致的任务复用，其余继续；异常不是有效失败回合，不能删seed或按部分输出计算正式分数。换模型、改并发设置或换release用新输出目录。比较baseline/FIXED时也用同一当前种子表及评测配置；不要把历史不同随机流/不同horizon的分数称作逐回合配对结果。

## 7. 本次交付的验收边界

本次测试对象是**8卡70k配置生成、真实launcher dry-run、恢复选择规则、正式GT拒绝规则、50×50/8卡评测命令**；验收凭证见 [PRI4R_8GPU70K_HANDOFF_VALIDATION.json](../provenance/PRI4R_8GPU70K_HANDOFF_VALIDATION.json)。方法计算、GT物理时钟及保存恢复另有此前单H100真实短训/推理证据：[物理时钟修复](PRI4R_PHYSICAL_CLOCK_FIX.zh-CN.md)。没有把CPU配置测试称为8张实体GPU训练通过，也没有70k训练或本版benchmark成绩。

远端保留：固定commit/manifest、两份环境freeze、全GT凭证、resolved YAML、完整训练run、最新全state及70k EMA导出、评测原始结果和FINAL_ACCEPTED。首次8卡真实运行暴露的环境/NCCL/资源问题按原始栈处理；不通过修改GT凭证、删rank状态或降低验收标准继续。
