# 给Astra：StreamPI迁移到RoboCasa365，8卡/global64/70k

这是EXP-007的主执行文档。用户已明确：**论文与公开代码冲突时以公开代码为准；原始π0.5预训练权重初始化；8卡，global batch64，训练至70000次优化更新。** 本文和配置是后续执行依据，不沿用C3、Pri4R、EventVLA或此前A/B的配方。

当前交付包含方法核心、完整RoboCasa读取器、单机8卡JAX FSDP训练入口和自动恢复。**实体8卡完整模型的显存、通信、吞吐、恢复尚待Astra验收；不要把CPU虚拟设备检查或旧单H100证据当成已通过8卡。** 先执行第4节，再配置正式训练。此交接以70k训练、完整状态和EMA导出为完成条件；没有授权本轮发布者启动70k，也没有新benchmark成绩。

## 1. 固定配方

| 项目 | 要求 |
| --- | --- |
| 官方来源 | hku-sail/StreamPI commit `ba024b2149ed94542a6c3352554f4f52f766a26d` |
| 模型 | 原始JAX π0.5，3,353,433,872参数全量训练；无新网络参数，无LoRA，无冻结VLM |
| 历史 | T3包括当前；固定`[max(t−10,0),max(t−5,0),t]`，jitter关闭；当前anchor可为所有有效帧 |
| Attention | 单元内图文双向、跨单元因果；当前VLM读取旧KV，累计有效token RoPE |
| 短历史训练 | 最早k个完整单元被mask，k均匀取0/1/2，整个global batch共用该标量；保持官方实现 |
| 在线缓存 | 编码前每T次调用整体清空，context为1,2,3,1…；只保存prefix KV，不存动作denoising KV |
| RoboCasa适配 | 三相机、16D state、逐时刻对应state文本；原生12D action，模型内部pad32；H50/执行5步 |
| 数据 | Human300，300任务×100演示=30000；任务权重frame_count^0.4；不继承A/B留出，不使用GT轨迹辅助数据 |
| batch/并行 | **单机8张GPU，一个Python进程，JAX FSDP8；global64，每卡8，acc1**。不要套torchrun，不启动8份本脚本 |
| 初始化 | 官方JAX `pi05_base/params`，不是PyTorch safetensors、社区30k、旧75k模型或工程step2权重 |
| 更新与日程 | **70000更新停止**；warmup10000、peak5e-5、decay_steps1000000、decay_lr5e-5，沿用官方模拟配置，不改为70k cosine |
| 优化器 | FP32 AdamW，β=.9/.95、eps1e-8、WD1e-10、clip1；EMA .999；BF16模型计算 |
| 保存 | 初始化step0、step1、每5000和停止点同步保存完整状态；停止点导出EMA、norm与配置；不自动删除旧checkpoint |
| 随机性 | seed42、训练key按全局update fold_in；数据按seed/sample index确定；每完成一次update游标增加64 |

GPU类型优先8×H10080GB。其它已有GPU类型必须先通过同配方8卡验收；不要按空闲卡数改成4卡，也不要把global64静默改小。代码暂只支持单机8卡，不把跨主机的8卡数量冒充受支持拓扑。

global64与70000停止点由用户指定，区别于作者原基准的batch/训练预算；“公开代码优先”不覆盖用户的这些明确要求。

每时刻state和H50是RoboCasa接口适配：官方未发布RoboCasa配方，不能直接搬LIBERO的H10/关闭state。方法核心与已核对版本相同；单H100工程代码的CPU optimizer offload改成了正式入口的native GPU FSDP，这项执行变化需实机验收。

## 2. 代码、目录和资源

服务器所有项目读写、缓存、临时文件和输出都必须在用户明确授权的工作区根目录内。下文`/YOUR_AUTHORIZED_ROOT`是占位符；先读取用户提供的本地部署补充说明，设置真实根目录，不能自行扩大权限范围。

使用独立checkout，固定本次交付commit；训练中不pull、不覆盖源码/配置。已有checkout先核对身份，不覆盖别人的任务。以下变量不涉及密码或token：

```bash
export ROBOT_WORKSPACE=/YOUR_AUTHORIZED_ROOT
export CODE_DIR="$ROBOT_WORKSPACE/robot-experiments-streampi"
# 仅在CODE_DIR不存在时clone；随后checkout到用户收到的本次发布commit。
git clone https://github.com/blacknwhit/robot-experiments.git "$CODE_DIR"
cd "$CODE_DIR"
git rev-parse HEAD
export POLICY_PY="$ROBOT_WORKSPACE/envs/existing-jax/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export STREAMPI_USE_HOST_DRIVER=0  # 如已验证环境需要主机驱动优先，可设1；仅改变本进程库搜索顺序
export RESOLVED="$ROBOT_WORKSPACE/streampi-resolved"
mkdir -p "$RESOLVED"
```

现有兼容模型环境：Python3.11、JAX/JAXlib0.5.3、Flax0.10.2、Orbax0.11.13、Optax0.2.8、Torch2.7.1+cu128。作者uv.lock的Optax0.2.4与此不同，已记录。优先复用，不为方便重装系统环境；授权目录内的兼容修复由Astra处理并记录，禁止系统安装、权限或网络安全设置修改。

复制 `experiments/exp007_streampi/resources.example.json` 到 `$RESOLVED/resources.json`，根据已有资源清单填写真实路径：

| 键 | 必需内容 |
| --- | --- |
| base_checkpoint | 原始官方JAX pi05_base的params目录 |
| norm_assets | RoboCasa已验证归一化统计目录；只读统计，不加载其它模型 |
| tokenizer | 已有paligemma_tokenizer.model |
| data_root | 包含robocasa365-pretrain-atomic和robocasa365-pretrain-composite的目录 |
| registry_file | 对应版本RoboCasa的dataset_registry.py |
| arrow_cache | 已有LeRobot v3 Arrow缓存目录 |

这些资源不会上传GitHub。新代码的数据读取器已包含在 `runtimes/streampi_reader/`，不依赖旧A/B工作目录。模型核心位于 `runtimes/streampi/`，执行代码位于 `experiments/exp007_streampi/`。

原始base目录树SHA256固定为 `737632cdece0b202ea7d35e32ad55a7c3f03f6c6606be41ed6c34d7996ca343f`。配置器会核验；如资源布局/元信息导致不匹配，先对比原始来源及逐文件哈希，不删掉检查强行加载其它模型。数据是现有LeRobot v3转换人口，task/source episode映射由读取器核对。

## 3. 发布后的本地检查与数据准备

CPU检查使用现有Python，8个**虚拟CPU设备**；不把它当作实体GPU验收：

```bash
"$POLICY_PY" experiments/exp007_streampi/test_cpu.py \
  --output "$ROBOT_WORKSPACE/zf-streampi-cpu8" --mode reference
"$POLICY_PY" experiments/exp007_streampi/test_cpu.py \
  --output "$ROBOT_WORKSPACE/zf-streampi-cpu8" --mode resume
```

输出目录必须新建、不覆盖既有测试。检查真实AdamW/EMA、FSDP数组分片、新进程step2→3恢复、丢弃不完整最新checkpoint及8卡/global64/70k配方约束。

接下来生成两个独立8卡smoke配置。它们的模型、batch、LR日程与正式70k一致，只用CLI提前停止，不缩短学习率日程：

```bash
for RUN_ID in zf-streampi-smoke-ref zf-streampi-smoke-resume; do
  "$POLICY_PY" experiments/exp007_streampi/prepare.py configure \
    --workspace-root "$ROBOT_WORKSPACE" --resources "$RESOLVED/resources.json" \
    --run-id "$RUN_ID" --mode smoke --out "$RESOLVED/$RUN_ID.json"
  "$POLICY_PY" experiments/exp007_streampi/prepare.py check-data \
    --config "$RESOLVED/$RUN_ID.json"
done
```

`configure`封存源码、原始base、norm/tokenizer、任务registry和数据元信息SHA；`check-data`核验300×100人口、4份可逐SHA复核样本，以及**真实global64的完整图像/token/action batch**。输入预处理为三相机`[64,3,224,224,3]`、prompt`[64,3,200]`、action`[64,50,32]`。不生成新GT，不改变评分。

配置输出拒绝覆盖；源码变化会使旧resolved config失效。需要修复时先保留失败日志，生成新的配置/独立测试目录，重新验收对应变化。不要在正在运行的目录改代码。

## 4. 实体8卡必须先通过短训与恢复

先记录GPU型号、驱动、软件版本、8卡可见性、显存和任务回执。仅设置8个可见GPU；一个进程管理全部8张卡。按如下顺序运行，**不要让多个8卡smoke在同一组卡上重叠**：

```bash
bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/zf-streampi-smoke-ref.json" --dry-run
bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/zf-streampi-smoke-ref.json" --stop-after 3

bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/zf-streampi-smoke-resume.json" --stop-after 2
# 上一进程确认结束后，用新进程继续；不改变源码或配置。
bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/zf-streampi-smoke-resume.json" --stop-after 3

"$POLICY_PY" experiments/exp007_streampi/compare_runs.py \
  --reference "$ROBOT_WORKSPACE/zf-streampi-smoke-ref" \
  --resumed "$ROBOT_WORKSPACE/zf-streampi-smoke-resume" \
  --out "$RESOLVED/GPU8_ACCEPTED.json"
```

必须看到：8张实体GPU、global64/FSDP8，真实完整模型更新且loss/各组梯度有限；恢复attempt从step2开始、数据cursor128；step3模型、Adam、EMA、RNG、cursor与连续运行逐值签名相同；完整检查点和EMA导出可读。`GPU8_ACCEPTED.json`仅由实体GPU比较生成，CPU测试不能代替。正式配置器会校验该凭证覆盖相同配方和代码。

同时记录初始化峰值、每卡显存、首编译耗时、稳定阶段step/data等待时间、GPU利用率、保存/恢复耗时与磁盘容量。三步只能验收功能，不能声称长训吞吐稳定；进入正式任务前可在该smoke配置进一步运行到20步观察，但保留step2/3检查点和凭证。对照签名检查只使用step3。

若OOM、通信失败或恢复不一致，由Astra定位并修复正常实现/兼容问题；保持8卡/global64、官方方法和数据协议。不要直接降低batch、冻结模块、切成CPU optimizer或把代码改成另一个方法。需要改计算组织时重新记录差异、完成验收；明确权限拒绝或配额拒绝则报告原始错误，不通过切环境绕过。

## 5. 配置并运行70k

8卡验收通过后，为正式训练新建独立run，从原始base开始，**不接着smoke权重训练**：

```bash
"$POLICY_PY" experiments/exp007_streampi/prepare.py configure \
  --workspace-root "$ROBOT_WORKSPACE" --resources "$RESOLVED/resources.json" \
  --run-id zf-streampi-70k --mode formal \
  --gpu-acceptance "$RESOLVED/GPU8_ACCEPTED.json" \
  --out "$RESOLVED/streampi-70k.json"
"$POLICY_PY" experiments/exp007_streampi/prepare.py check-data \
  --config "$RESOLVED/streampi-70k.json"
bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/streampi-70k.json" --dry-run
bash experiments/exp007_streampi/launch.sh \
  --config "$RESOLVED/streampi-70k.json"
```

入口采用单读取线程预取下一global batch，reader缓存不被多个线程并发修改；仅完成optimizer更新才提交数据cursor，预取内容在中断后可丢弃并重建。真实8卡运行中须测量数据等待。如果CPU读取成为瓶颈，Astra可以在保持索引、顺序、输入哈希及恢复语义的前提下优化；不得未经记录改变样本人口或随机协议。

当前每次完整状态约为数十GiB，初始化/step1/每5000/最终共约16份；另有EMA导出和两个smoke目录。正式训练前准备至少约1TB可用产物空间并按实测重估，不算数据本身。默认不删除旧checkpoint；不足时先报告真实容量，不删除其它实验。主存还需容纳初始化权重和数据预取，先实测。

## 6. SCO提交与自动恢复

普通训练/兼容修复已授权；Astra作为后续主执行聊天亲自负责修改、测试、部署和提交，不把编码交给子代理。本文不包含任何认证信息，也不要求创建新付费资源或提高配额。

- 使用已有SCO工具与当前 `sco-acp-job-launcher` 技能核实**单worker8GPU** Worker Spec。任务名统一`zf-xxx`，正式名建议`zf-streampi-70k`。
- 标准配额优先；标准实际拒绝/不可用后才转闲时。同样8卡请求可排队，不能仅因暂时空闲不足在本地等待或降卡。
- 提交必须设置并读回 **`autoRetrain=true`、`autoRetrainIntervalMinutes=10`**；记录native monitor ID、rootJob/currentJob、任务ID、启动命令、commit、resolved config、卡数、输出目录和真实启动日志。
- 任务启动命令只运行一次第5节的`launch.sh --config ...`，启动前设置本节需要的环境变量和正确cwd；不额外包8份torchrun，不使用Codex心跳作为训练恢复。
- 恢复时重复完全相同命令。入口选择最新完整且identity一致的checkpoint；校验文件SHA，恢复全量模型、Adam、EMA、step、随机key和样本游标；不完整保存跳过。文件锁防并发writer。
- 如果已有CLAIM但完全没有可恢复checkpoint，入口拒绝静默从base重跑。检查是否初始化阶段中断或产物损坏；保留原目录。必要时用新run-id重新开展初始化验收，而不是删除CLAIM掩盖失败。
- 如需定时只读检查，按项目约定复用独立GPT-6.1 Sol检查聊天，正常进度只记日志，新增异常/用户阻塞/关键验收去重汇报。SCO原生恢复仍是唯一恢复者；不要创建竞争重启循环。

## 7. 完成与交付

正式完成必须同时具备：

```text
zf-streampi-70k/FINISHED.json                 # step=70000, cursor=4480000
zf-streampi-70k/CONFIG.json
zf-streampi-70k/checkpoints/step_070000/     # Orbax全state + COMPLETE文件清单
zf-streampi-70k/exports/step_070000/         # EMA params、assets、config、COMPLETE
zf-streampi-70k/metrics.jsonl
zf-streampi-70k/attempts.jsonl
```

汇报固定commit、8卡验收记录、数据/资源身份、实际训练70000完成凭证、完整续训状态、EMA导出、loss曲线、耗时和异常处理。平台RUNNING/成功结束、只有loss、只有EMA权重都不能单独当作完成。70000是停止点，不把官方1M学习率日程重写成70k。

交接时没有新任务成功率。后续如要评测，使用本实验的JAX `policy.py` 和流式session生命周期接入既有RoboCasa协议；不能把JAX EMA直接交给旧PyTorch权重加载器，也不能把旧单H100短回路成绩当正式benchmark。评测工作不混入这次训练完成判据。

## 8. 已有证据与尚待完成

已核对固定官方源码和论文差异，已有单H100全参数训练、完整恢复、T3官方直接对照、短闭环证据；公开代码优先采样版本另完成一次全参数更新。它们说明方法和接口已有工程依据，但不是本次8卡生产入口的实机成绩。

本发布新增CPU8虚拟设备的实际FSDP/AdamW/EMA保存恢复检查、真实global64数据接入、启动契约与移植来源核验。包含任务/episode标识和数据哈希的原始回执保留在本地部署补充材料中，不在公开仓库分发；后续Astra应按本文命令生成自身的验收回执。正式8GPU运行、吞吐、长训稳定性与70k完成由Astra按本文继续完成；不要把“配置已写好”写成“8卡训练已验证”。

测试还发现并修复了初始化参数与EMA共享buffer时的重复donation：现有JAX0.5.3对已分片数组使用device_put(may_alias=False)仍可共享各shard存储，因此在初始化/恢复后使用Array.copy分离EMA。8个shard的非别名检查、开启donation的三步训练及新进程恢复均通过；没有改变EMA公式或增加模型参数。
