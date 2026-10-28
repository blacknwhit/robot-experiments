# EXP-004：3D位置轨迹辅助实验（A/B/C三组）

[返回全局实验总览](../../EXPERIMENTS.md) · [方法结构](METHOD.zh-CN.md) · [Query C说明](QUERY50.zh-CN.md)

这是独立于PI0.5基线、Query-FIXED和轨迹v2的**一次新实验，内部包含A/B/C三组对照**。默认在已经跑过PI0.5/FIXED的集群上复用已有资源。流程为：**核对并复用数据/环境/原始预训练权重 → 复用已核验GT或补生成GT → 三组独立训练至70k → 同一固定场景分别测试 → 本实验内汇总结果**。

“从base开始”指从同一份原始OpenPI `pi05_base`预训练权重微调，不是全模型随机初始化；也不是加载已经训练到70k的PI0.5/FIXED或旧轨迹权重。仅新增Query/辅助分支按配置初始化。

| 范围 | 本实验使用方式 |
|---|---|
| 公开数据、原始pi05_base、norm、tokenizer、Arrow | 版本和身份一致就直接复用，不重新下载/转换 |
| policy / simulator环境 | 依赖及导入路径检查通过就复用；仅为当前代码生成独立v3 overlay |
| 机械臂＋物体1024点GT | 已有且匹配则核验后复用；PI0.5/FIXED跑过本身不代表已有GT |
| 配置、日志、完整checkpoint、评测输出、执行记录 | 全部写入本实验独立根`E`；三组run-id仍分别独立 |
| 固定评测场景 | 同仿真版本、协议和计划身份一致可复用；否则为本实验生成一次，三组共用 |

已有环境请直接按第2节操作。**只有缺少资源或空白集群才查[按需准备/从零部署附录](FRESH_SETUP.zh-CN.md)**，不要把完整安装/下载命令当作每次新实验的必做步骤。

三个评测结果各2500计分回合，39个逻辑分片可分批排队、复用GPU。本文是操作手册；当前实际验收范围见第8节，本文更新没有提交新任务。

## 1. 三组分别是什么

| 项目 | A：`embedding3d_pos_a` | B：`embedding3d_temporal_b` | C：`query3d_temporal_c` |
|---|---|---|---|
| Z_t来源 | 原PI0.5观测prefix＋训练期独立Embedding | 同A | 50个Query加入VLM prefix，取attention后Query直接作为Z_t |
| Point Track Head | PointMLP＋fusion MLP | fusion后增加两层同点因果时间attention | 与B共用完全相同的点头 |
| 时间交互 | MLP逐时间输出 | 同一点读过去和自身，不跨点，不新增RoPE | 点头同B；上游Query仍按旧Query50规则交互 |
| 监督 | 完整50步相对起点位置L1＋相邻差分L1 | 同A | 同A |
| 动作推理 | 无3D输入，不执行点头/Embedding | 同A | 无3D输入，不执行点头；保留50个Query的prefix缓存 |

B/C时间块为width256、8heads、FFN512、pre-LN残差、dropout0。C保留旧非对称mask：观测只读观测，Query读观测及全部Query，动作读观测及Query；主干已有RoPE不变。详见[方法说明](METHOD.zh-CN.md)和[C结构说明](QUERY50.zh-CN.md)。三组、旧v2的checkpoint不能互换恢复，不能把B权重当成C的续训状态。

| 阶段 | 现成脚本 |
|---|---|
| 缺少资源时的环境准备 | `scripts/setup_policy.sh`、`scripts/setup_sim.sh` |
| 公开数据/基座 | `scripts/download.py`、`scripts/convert_base.py` |
| 训练子集/缓存 | `scripts/data_scope.py`、`scripts/prepare_arrow.py` |
| GT准备/分片生成/全集校验 | `scripts/prepare_gt.py` |
| 配置、训练、自动选续训 | `scripts/configure.py`、`scripts/train.py`、`scripts/train_or_resume.py` |
| 导入/GPU/NCCL/动作检查 | `scripts/import_preflight.py`、`scripts/doctor.py`、`scripts/nccl_probe.py`、`scripts/point_inference_probe.py` |
| 固定episode种子、分片评测 | `scripts/evaluate_seeded.py` |
| 单task评测/开发诊断 | `scripts/evaluate_seeded.py --task ...` |

## 2. 默认路径：复用旧集群资源，隔离本次实验

### 2.1 先定义三个目录和两个解释器

- `REPO`：本次实验的代码checkout。旧实验仍在运行时使用单独checkout，避免原地更新其源码；只克隆代码，不克隆大数据。
- `H`：**已有**PI0.5/FIXED公共资源根，沿用旧路径。例中使用原公开交接的目录布局。
- `E`：**新的独立实验输出根**，例如`/shared/experiments/exp004_trajectory_v3_abc`。
- `PY` / `SIM_PY`：实际已验证的policy与sim解释器绝对路径，可在`H/envs`，也可在别处。

```bash
set -euo pipefail
export REPO=/absolute/code/robot-exp004
export H=/absolute/existing/robot-data
export E=/absolute/experiments/exp004_trajectory_v3_abc
export PY="$H/envs/policy/bin/python"       # 改为已存在的policy Python
export SIM_PY="$H/envs/sim/bin/python"      # 改为已存在的sim Python
export ROBOT_EXPERIMENTS_HOME="$H"
# 仅当尚无本实验代码checkout时执行；已有专用checkout直接cd即可。
git clone https://github.com/blacknwhit/robot-experiments.git "$REPO"
cd "$REPO"
mkdir -p "$E/resolved" "$E/logs" "$E/receipts"
test -x "$PY" && test -x "$SIM_PY"
sha256sum -c provenance/source_files.sha256
printf 'export %s=%q\n' REPO "$REPO" H "$H" E "$E" PY "$PY" SIM_PY "$SIM_PY" > "$E/experiment.env"
cat >> "$E/experiment.env" <<'ENV'
export ROBOT_EXPERIMENTS_HOME="$H"
export OPENPI_DATA_HOME="$H/cache/openpi"
unset PYTHONPATH PYTHONHOME
cd "$REPO"
ENV
source "$E/experiment.env"
git rev-parse HEAD > "$E/receipts/code-commit.txt"
```

每个新作业/终端先加载本实验环境文件（替换为上面E的绝对路径）：

```bash
set -euo pipefail
source /absolute/experiments/exp004_trajectory_v3_abc/experiment.env
```

各GPU作业应能访问同一代码、公共资源和E目录。实验执行期间固定代码commit；已有run继续使用其原封存checkout/配置，不通过改旧配置迁移到E。新目录不要预先创建具体run目录，首次启动由训练器创建。

```text
H/                            # 旧公共资源，继续复用
  data/robocasa365/            # 视频、parquet等
  data/training_scope.json
  data/gt_inputs/              # 若曾做过轨迹GT
  data/gt_armobj_full_v1/      # 匹配时直接复用
  models/pi05_base_pytorch/model.safetensors
  models/robocasa_norm/
  cache/{arrow,openpi}/
  envs/{policy,sim}/           # 解释器也可位于其它已验证路径
  playground/Code/            # 已有RoboCasa/robosuite和资产
E/                            # 本次实验独立产物
  experiment.env
  resolved/ logs/ receipts/
  smoke_runs/ engineering_runs/ runs/
  eval/ evaluation-scenes/
```

### 2.2 核对数据与原始预训练权重，不重复下载

`H`应指向原资源根，而非新建的空数据目录。训练数据是300任务、30,000演示、27,815,381帧，不能换成全部32,043条聚合演示。直接核对已有scope和原始base/norm身份：

```bash
"$PY" - <<'PY'
import json, os
from pathlib import Path
h = Path(os.environ['H'])
d = json.loads((h/'data/training_scope.json').read_text())
assert (d['tasks'], d['episodes'], d['frames']) == (300, 30000, 27815381)
assert (d['filter_key'], d['filter_seed'], d['weight_alpha']) == ('100_demos', 0, 0.4)
assert d['registry_sha256'] == '23e8e31bb56bae80de86c34eafadfe32dd1e1eba6608376152ba9fdda5532a4b'
assert (h/'cache/arrow').is_dir()
assert (h/'cache/openpi/big_vision/paligemma_tokenizer.model').is_file()
print('Existing scope/cache paths found; retain and check their original download/metadata receipts.')
PY
printf '%s  %s\n' \
  9fd36e337832960bbfc122bd686dfc64b89551522d891c0d21ae586385de297e "$H/models/pi05_base_pytorch/model.safetensors" \
  4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee "$H/models/robocasa_norm/norm_stats.json" | sha256sum -c -
```

同时核对`receipts/download_datasets.json`的两个dataset revision及`training_scope.json`的metadata SHA是否对应现有数据。scope或Arrow缺失才执行附录中的对应准备命令；数据已有且身份一致就跳过下载。原始base已存在且SHA相同就跳过download/convert。**70k模型路径不能填入`framework.pi05.init_checkpoint`**。

如果旧集群不是本公开包的标准目录布局，先登记真实路径并按模板逐项映射`init_checkpoint/assets_dir/data_root_dir/v3_arrow_cache_dir/point_track_gt_root`；不要为了凑默认路径再下载一套。解析后的资源路径与恢复配置一经确定需冻结。GT还受原始asset_root身份约束，见第3节。

### 2.3 复用解释器，检查v3差异和导入路径

下面只读已有环境并记录到E，不运行`setup_policy.sh/setup_sim.sh`，不修改旧环境。policy目标为Python3.11、Torch2.7.1+cu128、Transformers4.53.2、NumPy1.26.4及既有训练依赖；sim为MuJoCo3.3.1、NumPy2.2.5及原固定RoboCasa/robosuite。完整版本参照[公共部署参考](../../HANDOFF.zh-CN.md)。

```bash
"$PY" - <<'PY'
import sys, importlib.metadata as m
assert sys.version_info[:2] == (3, 11)
for name, expected in [('torch', '2.7.1+cu128'), ('transformers', '4.53.2'), ('numpy', '1.26.4'),
                       ('accelerate', '1.5.2'), ('deepspeed', '0.16.9'), ('pyarrow', '25.0.0')]:
    actual = m.version(name)
    print(name, actual)
    assert actual == expected, (name, actual, expected)
PY
"$SIM_PY" -c "import mujoco,numpy; print(mujoco.__version__,numpy.__version__); assert mujoco.__version__=='3.3.1' and numpy.__version__=='2.2.5'"
uv pip freeze --python "$PY" > "$E/receipts/policy.current.freeze.txt"
uv pip freeze --python "$SIM_PY" > "$E/receipts/sim.current.freeze.txt"
cmp "$H/receipts/sim.freeze.txt" "$E/receipts/sim.current.freeze.txt"
# 只在本次代码checkout中建立trajectory_v3 overlay；A/B/C共用它。
"$PY" scripts/runtime_overlay.py --variant embedding3d_pos_a
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  "$PY" scripts/doctor.py --variant "$V"
done
```

若缺`uv`，只安装该工具，不重装训练环境。freeze文本不同先查明差异，不能覆盖旧`H/receipts/sim.freeze.txt`来冒充相同仿真；已有场景的身份也绑定此freeze。依赖确需补齐时，在本实验独立环境中补差异并更新PY/SIM_PY，保留旧环境；详见附录。原包overlay不能复制到全局site-packages，不能把旧baseline/FIXED的overlay或policy shim带入v3/真实模拟器。

第4节在GT就绪后还会用`import_preflight.py`核验实际lazy imports及模块来源。仅依赖/import通过不等于GPU训练验收；已有完全相同组合可复用对应证据，新增C按原范围做最小检查。GPU容器创建时需要`NVIDIA_DRIVER_CAPABILITIES=all`；在已缺图形驱动库的容器内export不能补挂载。

### 2.4 配置同时绑定公共资源和独立输出

后文每次configure都传`--home "$H" --experiment-root "$E"`。前者只提供数据/base/GT/缓存；后者只决定smoke/short/formal的输出根。训练仍使用H中的tokenizer缓存，运行配置记录两者。这个新选项仅用于本次A/B/C，不改变旧PI0.5/FIXED配置默认行为。

## 3. GT优先复用，仅缺失时生成，A/B/C共用

PI0.5/FIXED不使用这套GT，因此“那两个实验跑过”不能直接跳过检查。如果之前轨迹v2已生成1024点机械臂＋物体GT，目录、sampling_config、输入manifest、来源/帧对齐和数组身份均匹配，就复用它；**跳过prepare和generate，直接执行下方verify**。只读核验数组，成功后更新同一GT的验收凭据，不重新生成数组。

GT位于`$H/data/gt_armobj_full_v1`，`asset_root`保留原H；输出根E不参与GT哈希。若GT来自另一套绝对asset_root，不能改sampling_config或伪造marker来绕过身份校验，应先完成路径/来源兼容核查。

**仅GT缺失时**：确保现有数据scope、厨房资产和MuJoCo已就绪，按prepare → 8分片generate → verify顺序补齐新实验特有的GT。准备一次，A/B/C共用。以下前两个生成代码块已有GT时跳过。

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage prepare
# prepare只执行一次并等待成功；然后8个CPU作业各取一个I：0,1,2,3,4,5,6,7。
# 以下是“单个分片”的完整生成命令，每个作业先source `$E/experiment.env`。
I=0
"$PY" scripts/prepare_gt.py --home "$H" --stage generate \
  --workers 6 --shard-index "$I" --num-shards 8 2>&1 | tee "$E/logs/gt-shard-${I}.log"
```

没有CPU作业数组时，可在一个有足够内存的CPU作业中**顺序**执行下面循环。与上面的8作业方案二选一，不同时运行，不让两个进程写同一分片。每个分片6个workers；8个并行分片合计48个workers，按实际CPU/内存分配。

```bash
for I in 0 1 2 3 4 5 6 7; do
  "$PY" scripts/prepare_gt.py --home "$H" --stage generate \
    --workers 6 --shard-index "$I" --num-shards 8 2>&1 | tee "$E/logs/gt-shard-${I}.log"
done
```

**已有GT直接从这里开始；新生成GT须等全部8个分片成功后**，单独执行全集核验：

```bash
"$PY" scripts/prepare_gt.py --home "$H" --stage verify
cat "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
```

核验结果应为`episodes=30000`、`frames=27815381`、`verified_all_array_hashes=true`，并带`config_sha256`和输入manifest SHA。缺分片或损坏会报错；不要自行创建通过标记或跳到训练。分片失败时保留日志，修复原因后只重跑对应I，再重新全集verify。

生成器沿用已审计的`runtimes/trajectory_v2/starVLA/pri4r/`原始轨迹实现；A/B/C v3改变的是reader和监督，**不是重造另一套GT几何**。prepare按NVIDIA固定revision下载states/XML/metadata并核对原始与训练数据的state/action/frame/timestamp；generate重放记录状态，不积分预测动作。

1024个表面点：通常512机械臂＋512直接接触对象；无接触对象时1024机械臂；无背景。保留静止点，不用“运动大”筛选点。原始轨迹、刚体变换和有效性记录用于v3 reader：固定窗口首帧机器人base坐标，构造`p[t]-p[0]`和相邻差分；只监督连续有效前缀，缺帧/越界/尾部padding不当成零运动。GT未来点及mask只进loss。

资产路径为`$H/playground/Code/{robocasa365,robosuite}`，优先沿用旧实验已有安装和资产；仅缺失时按附录准备。`asset_root`属于配置SHA；新根重新生成时产生自己的SHA，configure会读取，不能硬填旧SHA。verify检查全部30,000个episode的来源、范围、数组哈希，写`data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json`；训练入口要求该凭据。

已有自有GT可整体搬运，须保留完整metadata/manifest/sampling_config并重新核验；不能只拷坐标数组或手写通过标记。旧完整训练的恢复还绑定旧代码与配置身份，本页新发布代码不支持假装与旧SCO运行逐位兼容；旧运行继续使用其封存release。

## 4. 三组首跑的最小训练/恢复检查

这是**新集群首次运行或运行组合改变时**的短检查，不是重复全量评测。已经在完全相同代码/环境/入口组合验证的项目可复用证据。C目前尚无真实GPU验收，首次上新集群须先确认可训练/保存/恢复/推理，不能用A/B历史结果代替。

GT通过后，一次性为三组生成各自steps2/3配置：

```bash
test -f "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  for STOP in 2 3; do
    "$PY" scripts/configure.py --variant "$V" --home "$H" --experiment-root "$E" --run-id "${V}-smoke" \
      --gpus 2 --smoke --stop "$STOP" --out "$E/resolved/${V}-smoke${STOP}.yaml"
  done
done
```

在实际分配的2×H100作业里执行以下短流程。A/B/C分别令`V`为表中配置名；可使用三个独立作业，或在同一双卡作业内逐组执行。每组均B32×2、acc1、global64。

```bash
V=embedding3d_pos_a  # 另两组分别为 embedding3d_temporal_b / query3d_temporal_c
"$PY" scripts/import_preflight.py --config "$E/resolved/${V}-smoke2.yaml"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/nccl_probe.py
"$PY" scripts/train.py --config "$E/resolved/${V}-smoke2.yaml"
# 另一个Python进程完整恢复step2，再执行step3：
"$PY" scripts/train.py --config "$E/resolved/${V}-smoke3.yaml" --resume
```

检查实际GPU型号、两rank NCCL、steps1/2完整保存、新进程恢复后step3；每rank FP32主权重/Adam/BF16/RNG审计、有限loss/gradient、共享VLM/点头以及A/B的Embedding或C的future_queries更新；B/C还检查两层时间块。仅import通过不算GPU通过。完整checkpoint需要模型、两rank Adam/RNG、scheduler/data clocks、trajectory contract及`PORTABLE_COMPLETE.json`；残缺的新checkpoint跳过，没有完整状态则报错。

训练smoke自动保存`engineering_sample.pt`，用自己的该样本运行：

```bash
RUN="$E/smoke_runs/${V}-smoke"
"$PY" scripts/point_inference_probe.py --config "$E/resolved/${V}-smoke3.yaml" \
  --checkpoint "$RUN/checkpoints/steps_3_model.safetensors" \
  --sample "$RUN/engineering_sample.pt" --output "$RUN/action_probe.json"
```

它验证完整权重严格加载、无3D输入及joint/cache一致性。下面高效闭环入口另会严格加载所有模型key并禁止点头执行。

## 5. 三组独立训练至70k

三组都从**同一个原始pi05_base预训练权重、seed42**独立开始；不从其他组、旧Query、smoke或短训权重初始化。先一次性生成三份正式配置，并在CPU上dry-run确认命令（不会启动训练）：

```bash
test -f "$H/data/gt_armobj_full_v1/PORTABLE_GT_ACCEPTED.json"
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  "$PY" scripts/configure.py --variant "$V" --home "$H" --experiment-root "$E" --run-id "${V}-70k-s42" \
    --gpus 2 --formal --stop 70000 --out "$E/resolved/${V}-70k.yaml"
  "$PY" scripts/train_or_resume.py --config "$E/resolved/${V}-70k.yaml" --dry-run
done
```

三个训练作业分别申请**2×H100**。每个作业先source `$E/experiment.env`，然后只执行自己对应的一条命令；资源不足可以排队或顺序跑，不能三条同时占用同一双卡分配：

```bash
# A作业
"$PY" scripts/train_or_resume.py --config "$E/resolved/embedding3d_pos_a-70k.yaml"
# B作业
"$PY" scripts/train_or_resume.py --config "$E/resolved/embedding3d_temporal_b-70k.yaml"
# C作业
"$PY" scripts/train_or_resume.py --config "$E/resolved/query3d_temporal_c-70k.yaml"
```

| 组 | 独立训练目录 | 70k评测权重（相对训练目录） |
|---|---|---|
| A | `$E/runs/embedding3d_pos_a-70k-s42` | `checkpoints/steps_70000_model.safetensors` |
| B | `$E/runs/embedding3d_temporal_b-70k-s42` | 同名，但必须取B目录 |
| C | `$E/runs/query3d_temporal_c-70k-s42` | 同名，但必须取C目录 |

三组GT目录和`point_track_gt_config_sha256`相同；run-id、配置、checkpoint和日志分别独立。确认70k权重旁边的`checkpoints/checkpoint-70000/PORTABLE_COMPLETE.json`已生成，再交给评测。

没有`--formal`不会把v3默认带入长训。首次启动和每次调度重试均使用上面相同命令；不要在重试脚本重新configure或改run-id。`train_or_resume.py`首次新run启动base，run一旦存在就要求完整恢复；没有“恢复失败重新base”的兜底。恢复只能用相同world、相同代码/recipe，且持有唯一writer锁；本手册不支持2卡直接恢复到8卡。更新代码后不能继续拿旧resolved配置冒充旧release。

A/B/C训练子进程在导入Torch前固定`CUBLAS_WORKSPACE_CONFIG=:4096:8`。使用SCO时设置并核验`autoRetrain=true`、`autoRetrainIntervalMinutes=10`，记录monitor/root/currentJob；标准实际拒绝后转SPOT，保持两卡排队。其他集群由其调度器作为唯一重试owner，不再启动竞争恢复循环。

冻结候选：seed42、G8 ordered_random、global64/acc1、AdamW β0.9/0.95、eps1e-8、weight_decay1e-10、clip1、LR峰值2.5e-5/最低2.5e-6、warmup1000、scheduler时域100000，70k只是停止点。总loss为`action + ramp×(position + difference)`，辅助目标权重1.0、差分系数1.0、ramp前1000update线性升至1。这是当前工程候选，**没有证明该辅助量级最优**。首次update1额外完整保存，以后每1000和最终stop保存。

短训/正式都检查最初三条有限指标、预热40步后两个非重叠100步速度窗口和首个1k完整状态。loss下降不代表闭环效果提升。保留配置、freeze、日志、恢复审计和完整checkpoint。

若还需要先观察1500步训练动力，使用下面**可选**的独立短训；无需每次全量重做。令V为A/B/C中的目标组，在对应双H100作业执行。短训完成后正式70k仍从base开始，不把短训checkpoint混入正式run：

```bash
V=query3d_temporal_c
"$PY" scripts/configure.py --variant "$V" --home "$H" --experiment-root "$E" --run-id "${V}-short-s42" \
  --gpus 2 --engineering --stop 1500 --out "$E/resolved/${V}-short.yaml"
"$PY" scripts/train_or_resume.py --config "$E/resolved/${V}-short.yaml"
```

## 6. 三组使用统一task/episode种子直接评测

当前统一使用`evaluate_seeded.py`和`fixed_episode_seeds_v1`。每GPU一个task、一份模型服务、一个主线程管理多个独立环境的仿真进程；网络线程异步请求同一服务，模型保持batch1、原精度、10次去噪、预测50步、每5步重规划。实际环境与运行方式见[远端Astra执行说明](../../docs/ASTRA_FIXED_SCENES.zh-CN.md)。

A/B/C都读取仓库内同一任务清单和episode种子定义，不生成或读取PLANS，不比较预保存的图像/state/轨迹。每个task固定episode0–49，构造环境前使用对应seed；策略噪声仍用base7、task、逻辑环境0、全局episode和replan编号确定。不要改成每个worker固定seed7/8/9/10。

```bash
"$PY" scripts/evaluate_seeded.py --check-only
```

应显示50任务、2500计分回合、`reference_plan_required: false`。该命令不创建环境或加载模型。任务清单已有horizon、成功条件不改，horizon不再乘1.5。此新协议与旧391/559等历史成绩不是逐场景配对，历史分数单列。

每组70k完成后分别评测，不必等待另外两组。每组parts1–12各含4个task、申请4卡；part13含2个task、申请2卡，合计50个task，每个task一张GPU、50回合。

| 组 | 每个评测作业的V | 该组共用输出目录（默认n4） |
|---|---|---|
| A | `embedding3d_pos_a` | `$E/eval/embedding3d_pos_a-70k-seed-only-v1-n4` |
| B | `embedding3d_temporal_b` | `$E/eval/embedding3d_temporal_b-70k-seed-only-v1-n4` |
| C | `query3d_temporal_c` | `$E/eval/query3d_temporal_c-70k-seed-only-v1-n4` |

下面是一个分片作业的完整命令。先source `$E/experiment.env`，选择V和PART；同组所有分片使用同一OUT、N_ENVS与权重，组间OUT独立。

```bash
V=embedding3d_pos_a   # B: embedding3d_temporal_b；C: query3d_temporal_c
PART=1               # 每个独立分片分别为1..13
export N_ENVS=4      # 不是其它模型的容量或利用率保证
CONFIG="$E/resolved/${V}-70k.yaml"
CHECKPOINT="$E/runs/${V}-70k-s42/checkpoints/steps_70000_model.safetensors"
OUT="$E/eval/${V}-70k-seed-only-v1-n${N_ENVS}"

test -s "$CHECKPOINT"
test -s "$E/runs/${V}-70k-s42/checkpoints/checkpoint-70000/PORTABLE_COMPLETE.json"
if (( PART >= 1 && PART <= 12 )); then
  GPU_IDS=0,1,2,3; EGL_IDS=0,1,2,3
elif (( PART == 13 )); then
  GPU_IDS=0,1; EGL_IDS=0,1
else
  echo "PART必须是1..13" >&2; exit 1
fi
COMMON=(--variant "$V" --home "$H" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --sim-python "$SIM_PY" --output "$OUT" --n-envs "$N_ENVS" --timeout 43200)
"$PY" scripts/evaluate_seeded.py "${COMMON[@]}" --part "$PART" \
  --gpu-ids "$GPU_IDS" --egl-device-ids "$EGL_IDS"
```

GPU编号须在作业内与EGL物理UUID对应；重映射时传实际编号。平台作业时限应覆盖每张卡分配的任务；`--timeout 43200`只给一个task全部50回合12小时墙钟预算，不缩短horizon，也不改变平台自身回收规则。同机共享网络时使用独立`--port-base`。

普通同接口权重评测无需每task/每权重先做完整smoke。正式入口严格加载权重，首个真实请求核对eager/Graph完整预测动作，后续保留有限动作、三路RGB/状态健康和episode覆盖检查；失败保存日志并停止，不把异常计作正常失败分数或跳过回合。开发者有限诊断参数见当前Astra说明，不产生2500回合正式验收。

`n_envs`接口支持1–50；默认4不代表所有模型或task的容量已测。4卡n4配置最多4个仿真进程、16个独立环境。更改权重、配置、代码、依赖或n_envs使用新OUT，继续使用同一seed定义。无需叠加tasks-per-gpu或私有RGB补丁。推理不执行3D点头，C保留Query；基线的显存/速度不能冒充C实测。

失败后保留原日志；排除原因后重跑同一V/PART，入口只复用完整且身份匹配的task。不能同时启动同一task的两个writer。没有全部50个task时汇总拒绝生成完整成绩。

## 7. 汇总并比较三组结果

每组完成后可独立汇总；下例用于三组都完成后，不加载模型、不需要GPU：

```bash
export N_ENVS=4
for V in embedding3d_pos_a embedding3d_temporal_b query3d_temporal_c; do
  "$PY" scripts/evaluate_seeded.py --variant "$V" --home "$H" \
    --config "$E/resolved/${V}-70k.yaml" \
    --checkpoint "$E/runs/${V}-70k-s42/checkpoints/steps_70000_model.safetensors" \
    --sim-python "$SIM_PY" --output "$E/eval/${V}-70k-seed-only-v1-n${N_ENVS}" \
    --n-envs "$N_ENVS" --summarize
done
```

成功凭据是三个独立`FINAL_ACCEPTED.json`：各50份原始Boolean成功数组、2500个唯一task/episode组合，及相符的模型/config/source/seed/模拟器身份。平台SUCCEEDED或目录数量不能替代它，也不需要旧的episode轨迹hash。打印三组结果并核对共同协议：

```bash
"$PY" - <<'PY'
import json, os
from pathlib import Path
experiment = Path(os.environ['E'])
n_envs = int(os.environ['N_ENVS'])
cohorts = set()
for group, variant in [('A', 'embedding3d_pos_a'), ('B', 'embedding3d_temporal_b'), ('C', 'query3d_temporal_c')]:
    path = experiment / 'eval' / f'{variant}-70k-seed-only-v1-n{n_envs}' / 'FINAL_ACCEPTED.json'
    d = json.loads(path.read_text())
    identity = d['identity']
    assert d['trials'] == 2500 and len(d['tasks']) == 50
    assert identity['variant'] == variant
    assert d['episode_protocol'] == 'fixed_episode_seeds_v1'
    cohorts.add((identity['scene_seed_definition_sha256'], identity['task_registry_sha256'],
                 identity['source_manifest_sha256'], identity['sim_freeze_sha256']))
    print(f"{group}\t{variant}\t{d['successes']}/{d['trials']}\t{100*d['successes']/d['trials']:.2f}%")
assert len(cohorts) == 1, '三组任务/种子/代码或模拟器环境不一致，先核对运行身份'
PY
```

保留每task成功数组、episode编号、seed、错误证据及50项分项。三组各2500回合，共7500个计分回合；开发诊断不计入。旧分数按原协议单列；若以后需要同种子基线/FIXED对照，再独立安排相应权重的评测，不自动重测旧模型。

## 8. 已验证与未验证

2026-10-24的C结构增补：7项定向CPU检查、2项A/B相邻检查及GitHub CPU CI通过；A/B父版本初始化state、RNG、输出一致。C尚无真实GPU训练/恢复/闭环成绩。本手册给出三组完整操作过程，不代表这些操作已在新集群执行。C证据见[定向检查记录](../../provenance/QUERY50_TEMPORAL_C_VALIDATION.json)。

2026-10-23本次实测：

- Windows 68项检查中64通过、4项Linux专属跳过；Linux 68项全部通过。
- 单H100上A1500严格加载861keys、Bsmoke3严格加载885keys，分别完成n_envs=1/2各两个完整900步回合。合计8回合/7200模拟步/1440次完整50步动作对照；eager与graph逐元素完全一致，单/双环境的初始化与完整轨迹一致，CUDA与EGL UUID一致。
- 真实OpenDrawer演示204帧/1024点，在隔离目录重新生成8个数组，与原GT逐字节一致。复用历史环境时曾发现policy Python未装MuJoCo，因此该有界重建使用已安装MuJoCo3.3.1的sim Python；新安装脚本已包含该依赖，上方import检查不能省略。另一个缺少alignment凭据的旧样本输入被正确拒绝，随后使用真实既有凭据及实际源/缓存SHA核验后的输入，没有放宽校验。
- 最终训练启动代码核验961份源码；旧release配置会被拒绝；通过真实CUDA确定性矩阵前向/反向环境探针与正式配置dry-run。这不是新增完整模型训练结果。
- 补充的训练启动保护没有改变任何模型、GT或评测源码；最终发布与上述GPU运行版本的这些文件已逐项证明相同。

高效后端的正确性通过；本次smoke逐请求同时跑eager对照，**不是生产模式吞吐测试**。没有把baseline的12/24环境速度或利用率当作A/B性能，也没有据此宣称算法效果提升。

本次发布的精确源码和实际A/B加速验证见[发布验收记录](../../provenance/POSITION_V3_HANDOFF_VALIDATION.json)。此前A1500、Bsmoke3的训练/完整恢复/动作推理、fslgpu eager闭环已经通过；这些历史证据不能自动替代本次新图后端的验收。

原SCO ACP曾发生NVIDIA EGL内部SIGABRT；容器capabilities=all也不能证明此问题已消失。fslgpu通过只对实测组合有效。新集群若出现原生崩溃、初始化SHA不一致或动作对照失败，保存任务身份/环境/栈并回交主对话，不自行修模型、改成功条件或删坏场景。

GitHub不含数据、GT大数组、权重和原训练完整状态。代码整理与有界GPU验收不等于在新集群从下载开始重跑全部数据和70k实验，也不等于方法性能提升。

`deployment/sco_position_v3_debug/`保留旧集群排障取证，不是新集群启动入口；新集群只使用本页的scripts/configs。
