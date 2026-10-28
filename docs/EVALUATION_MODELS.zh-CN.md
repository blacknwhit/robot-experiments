# 多版本模型如何共用固定种子评测

统一使用 [8×H100高吞吐执行说明](ASTRA_FIXED_SCENES.zh-CN.md) 的 `scripts/evaluate_parallel.py`。场景、固定2500个episode种子、并行物理、EGL渲染、任务分配和计分共用；每一版使用自己的模型代码、配置、权重、归一化和输出目录。

## 当前已登记的模型

下面的“已登记”表示CLI、runtime、配置与服务端结构检查已有接线，**不等于每个版本都完成了最新parallel入口的GPU验收**。

| 版本 | `--variant` | 模型源码目录 | GPU证据边界 |
|---|---|---|---|
| PI0.5基线 | `baseline` | `runtimes/baseline` | 先前parallel全50task、100完整回合；本次adapter完整/短回合验证见文末 |
| 最佳Query-FIXED | `query_fixed` | `runtimes/query_fixed` | 旧fixed-scene有H1008回合/2880请求对照；本次外部adapter完整/短回合通过，范围见文末 |
| Query3D-v2 | `query3d_v2` | `runtimes/trajectory_v2` | 有历史70k评测；尚无新版parallel专属GPU验收记录 |
| Embedding3D-v2 | `embedding3d_v2` | `runtimes/trajectory_v2` | 有历史70k评测；尚无新版parallel专属GPU验收记录 |
| v3 A：完整相对位置＋MLP点头 | `embedding3d_pos_a` | `runtimes/trajectory_v3` | 旧handoff用A1500权重完成H100 n1/n2闭环与eager/Graph对照 |
| v3 B：同一点时间attention | `embedding3d_temporal_b` | `runtimes/trajectory_v3` | 旧handoff用B工程step3权重完成H100 n1/n2闭环与eager/Graph对照 |
| v3 C：Query50＋时间点头 | `query3d_temporal_c` | `runtimes/trajectory_v3` | 仅CPU结构/接线证据；未证明完整模型GPU加载或闭环已通过 |

证据分别见 [最新parallel验收](../provenance/PARALLEL_EVALUATION_VALIDATION.json)、[FIXED历史验收](../provenance/FIXED_SCENE_VALIDATION.json)、[A/B历史验收](../provenance/POSITION_V3_HANDOFF_VALIDATION.json)、[C的验证边界](../provenance/QUERY50_TEMPORAL_C_VALIDATION.json)。历史训练和成绩由 [实验总览](../EXPERIMENTS.md) 管理；不能把工程step3/1500当70k模型，也不能将旧协议分数混入新种子评测。

**最早的Query50/Pri4R Embedding相邻位移v1没有独立登记。** 不要将它们的旧权重直接标成`query3d_v2`或`embedding3d_v2`。v2里的结构名仍可能带`_v1`，那是模块名称，不是整个实验版本；历史v1的`adjacent_v1`目标与v2的`adjacent_and_cumulative_v2`不同。虽然后者的底层代码保留旧目标分支，仍需核对原始训练配置/源码/权重，登记独立身份后才能声称适配完成。更早Memory等结构也不在上表覆盖内。

## 对已有版本，操作上只换四项

例如评测Query-FIXED，按Astra文档执行相同8卡命令，只替换：

```bash
VARIANT=query_fixed
CONFIG=/absolute/path/to/this-model-resolved.yaml
CHECKPOINT=/absolute/path/to/this-model/steps_70000_model.safetensors
OUT="$HOME_DIR/eval/query-fixed-70k-parallel-seeds-v1"
```

换A/B/C或v2时，将VARIANT改为表中的对应值，使用该版本自己的config和checkpoint。配置必须保留实际训练时的结构和预后处理；迁移只修正资源/源码路径，不能用另一个版本的默认模板替代原训练定义。动作归一化资产必须与该模型对应。仅有相同tensor形状不足以确认语义相同。

每版一个独立OUT，在同一组8卡上顺序评测各版。task之间复用该版常驻模型；切换结构时启动新的策略服务，避免Python模块缓存或模型状态串版。任务/种子文件、原horizon及评分协议保持相同。完整回合结果由入口汇总，不手工拼接不同模型的raw JSON。

同一已接入结构后续换兼容checkpoint，无需重新生成种子/场景计划或重做整套50任务流程验收。入口仍会执行完整权重加载、首个真实请求eager/Graph对照、观测/动作健康及回合覆盖检查。不同结构的速度、显存和利用率需按实际运行记录，不能套用baseline数字。

## 新版本使用模型描述文件，评测代码保持固定

`evaluate_parallel.py`现在还接受 **`--model-spec /path/model.json`**，与`--variant`二选一。新模型ID不需要加入`common.py`，也不需要增加`server.py`分支。模型在独立策略进程中导入；仿真固定使用评测仓库baseline的客户端/仿真接口，不加载新模型的Python包。模型源码checkout与评测checkout可以分开维护。

同一家族换checkpoint时，复用描述文件，在评测命令中指定`--checkpoint /new/weights.safetensors`并使用新OUT即可。改变结构时，更新模型自身源码、训练配置和描述文件。原来的7种`--variant`调用仍兼容。

### 1. PI0.5兼容模型：复用内置适配器

模型仍提供StarVLA原生`build_framework(config)`、`policy_model`、`predict_action()`和相同预后处理时，可直接使用`factory=pi05`。内置适配器不规定Query数量、点头层数或RoPE开关；这些由模型训练配置和自身构建代码决定。它检查完整权重加载、环境动作空间，以及PI0.5的50步预测/12维动作/16维状态/10步去噪。模型负责保证训练辅助头不在动作推理中执行。

已有仓库版本可以生成描述文件：

```bash
"$POLICY_PY" scripts/create_model_spec.py \
  --model-id my-fixed-70k --variant query_fixed \
  --config /absolute/fixed-resolved.yaml \
  --checkpoint /absolute/steps_70000_model.safetensors \
  --policy-python "$POLICY_PY" \
  --asset normalization=/absolute/model-assets \
  --asset tokenizer=/absolute/paligemma_tokenizer.model \
  --output /absolute/model-specs/my-fixed.json
```

自己的模型checkout使用`--source-root`与`--python-path`替代`--variant`。可重复传入这两个参数，将模型、OpenPI源码及原先需要的兼容导入目录全部声明进去；路径次序就是导入次序。`--native`使用原生预测函数，默认使用已有PI0.5 CUDA Graph流水线。该流水线仍在首个实际请求自动核对eager/Graph动作，不要求另跑一套smoke。

### 2. 不兼容PI0.5内部接口：模型仓库增加一个adapter

适配器属于模型代码，**不用修改评测仓库**。描述文件指定`factory: "my_model.eval_adapter:create"`，调用约定为：

```python
from robot_eval.api import ModelAdapter

def create(context):
    # context: config, checkpoint, device, options, assets, backend等。
    # 用本模型代码构建、完整加载权重、装配原生预后处理。
    return MyAdapter(context)

class MyAdapter(ModelAdapter):
    # 默认stateless、managed_global RNG、max_concurrency=1、native后端。
    def predict(self, request, state):
        # request.images: 左相机/手腕/右相机，三个uint8[256,256,3]
        # request.state: float32[16]；request.language: str
        # request.seed: 固定task/episode/replan派生的uint32。
        # request.episode/replan仅供路由和记忆管理，不是新增模型输入。
        ...  # 返回本模型反归一化后的np.float32[50,12]环境动作。
```

这是接口示意，不是可以直接运行的空模型。完整类须提供`load_proof`：真实执行严格加载后产生的`strict/all_keys_present/all_loaded_values_exact_at_model_dtype/checkpoint_keys_dropped`证明。PyTorch完整safetensors可复用`strict_state_loader.strict_full_state_load(model, checkpoint)`；LoRA/多组件模型需由适配器完整加载其基座和各组件，所有文件列入`assets`，不能直接填写“已通过”常量。

模型自身处理resize、归一化、tokenizer、反归一化、采样器，构建后调用`eval()`，预测使用自己的`torch.inference_mode()`或等价原生推理路径。环境动作顺序固定为末端位置3＋旋转3＋夹爪1＋底盘4＋控制模式1；禁止把标准化latent或超过12维的动作交给评测端截断。输出只接受有限float32[50,12]，仍执行前5步再重规划。内部网络和去噪次数可以不同，通用接口不依赖PI0.5采样器。

当前SDK采用PyTorch严格数值设置：关闭torch.compile和TF32，保持模型原生混合精度，不全局转BF16。可用原生推理或模型自己实现的CUDA Graph后端；仅声明一个新backend名字不会自动启用加速。非PyTorch运行时或必须启用torch.compile的后端需单独扩展/验证数值执行契约，不能冒充本接口已验证。

### 3. 带历史记忆的模型

设置`state_mode="per_episode"`，实现`begin_episode(context)`返回**独立状态对象**，`predict(request, state)`只读写这个回合的状态，`end_episode(context, state, reason)`释放它。服务端实际传输begin/infer/end/abort事件，断连也清理该连接的回合；不使用全局`model.reset()`清空其他环境记忆。入口检查请求顺序、重复回合和同一个顶层状态对象被跨连接复用。不同容器内部共享的Tensor/KV缓存、模型全局memory仍由适配器负责隔离，需用交错请求验证，不能仅靠对象ID检查证明。

默认`rng_mode="managed_global"`串行执行模型调用，在`begin_episode`和`predict`前后保存/恢复Python、NumPy及PyTorch RNG；初始化记忆也有确定seed。清理函数不应改变模型的推理行为。自己管理局部Generator的模型可声明`explicit_per_request`，但必须真正使用`request.seed`。只有模型已实现线程安全、独占GPU缓冲区管理时，才提高`max_concurrency`（1–16）；基于现有PI0.5流水线的内置适配器为16。不要为了利用率给普通模型随意填16。

### 4. 描述文件和正式命令

外部adapter描述文件示例：

```json
{
  "format": "robot_eval_model_spec_v1",
  "model_id": "my-new-model-70k",
  "factory": "my_model.eval_adapter:create",
  "config": "/models/my-model/inference.yaml",
  "checkpoint": "/models/my-model/steps_70000_model.safetensors",
  "policy_python": "/envs/my-model/bin/python",
  "python_paths": ["/code/my-model"],
  "source_roots": [{"path": "/code/my-model", "include": ["**/*.py"]}],
  "assets": {"normalization": "/models/my-model/assets"},
  "backend": "native",
  "backend_version": "model_native_v1",
  "state_mode": "stateless",
  "rng_mode": "managed_global",
  "max_concurrency": 1,
  "options": {},
  "environment": {}
}
```

`source_roots`覆盖模型和adapter实际代码；额外配置、归一化、tokenizer、LoRA基座或自定义二进制也应在`include`或`assets`明确列出。入口记录这些文件和策略Python依赖的SHA身份，保存`MODEL_MANIFEST.json`，运行前后检查变化。不会自动推断任意模型代码打开的全部文件；不要漏列外部资产，不在描述文件放凭据。环境中的GPU编号、种子和数值协议由评测入口控制。

```bash
"$POLICY_PY" scripts/evaluate_parallel.py \
  --model-spec /absolute/model-specs/my-model.json \
  --home "$HOME_DIR" --sim-python "$SIM_PY" --output "$OUT" \
  --all-tasks --gpu-ids 0,1,2,3,4,5,6,7 \
  --egl-device-ids 0,1,2,3,4,5,6,7 --port-base 24090 \
  --n-envs 16 --tasks-per-gpu 50 --simulator-cpu-threads 1 \
  --rgb-schedule final --initial-rgb deferred --timeout 172800
```

其余资源、8卡/EGL映射、结果读取完全沿用Astra执行说明。一个入口负责50task×50回合；2500个种子不会因模型名、GPU、n_envs或调度次序改变。不要在运行期间改模型源码/配置/权重。不同模型性能和显存不同，高吞吐调度可以复用，但不能保证所有结构都达到baseline利用率。

## 新结构首次接入的最小验证

由主对话使用真实权重和少量真实观测，验证完整加载、输出动作形状/范围/有限性，以及同一输入和请求seed下原生推理与加速推理是否一致。再检查不同task/episode请求交错时没有状态或随机数串扰；必要时完成少量闭环确认接口。这里检查的是新模型与后端的连接，不是再生成/核对2500张参考图片，也不要求每个checkpoint重复全50任务验收。

没有真实权重、GPU加载或原生/加速结果时，只能标记“接线/CPU检查完成”。不能用baseline的流程验收替代C或未登记历史版本的模型侧证据；失败应保留日志交回主对话，远端不自行改结构、删参数或跳过出错回合。

## 2026-10-29：模型接口和单H100短测验收

真实PI0.5 baseline70k与最佳Query-FIXED70k均通过新适配器：实现阶段各完成2task×2完整回合、各720请求；最终发布代码又各检查4task×4seed的短闭环，并完成下表并发对照。FIXED由模型目录的外部factory接入，没有新增评测版本枚举。旧`--variant`实际20请求兼容性测试通过。Linux120项中119通过、1跳过；分阶段源码身份与原始证据SHA见[验收记录](../provenance/MODEL_ADAPTER_VALIDATION.json)。这些是接口/工程验证，不是新的模型benchmark分数。

跨模型**16/16对**初始三路RGB、机器人状态数组、MuJoCo状态hash及RNG状态hash全部一致。范围为CloseBlenderLid、CloseFridge、CloseToasterOvenDoor、CoffeeSetupMug各episode0–3；baseline在n_envs4/8/16的相同16个初始观测也逐值一致。未逐一重跑比较2500场景，不外推跨驱动逐像素保证。正式评测仍只读取固定种子定义，没有恢复参考图片/轨迹hash门槛；不同模型的后续动作及轨迹允许不同。

以下三档性能对照**仅使用PI0.5 baseline70k**；FIXED完成了模型接口与场景对照，不套用此表速度/利用率。单H10080GB、22vCPU、240GiB，模型常驻且共用一次预热；每档4task×4episode×40请求=640次策略请求。每次请求仍输出50步动作、执行5步，本诊断截短回合，不能计入正式成功率。

| n_envs | 整段仿真耗时 | 请求/秒（含初始化） | 整段GPU-util | 首请求→末步GPU-util |
|---:|---:|---:|---:|---:|
| 4 | 171.25秒 | 3.74 | 19.45% | 26.17% |
| 8 | 126.59秒 | 5.06 | 27.28% | 49.19% |
| 16 | 115.53秒 | 5.54 | 30.49% | 89.75% |

**所测档位建议n_envs16。** 比n4整段用时减少约32.54%，吞吐约1.48倍；比n8用时减少约8.74%。策略OMP/MKL=4、OpenBLAS=1，模拟器CPU线程=1，tasks-per-gpu50、final RGB和deferred初始化。单卡命令沿用上文，仅把两组GPU编号均改为`0`。这不证明16是所有CPU/模型上的全局最优；换结构后无需改评测，但速度及显存仍以实际测量为准。

三档实测峰值显存分别12536/16194/23743MiB（n16约23.2GiB）。按约1秒采样，三档最大采样间隔分别1.23/1.22/1.44秒，未丢弃低利用率样本，采集无错误；均值是nvidia-smi GPU-util采样算术均值，不是SM occupancy。

表中整段包括仿真进程启动、16个场景初始化、短rollout和收尾；**不包括共享模型加载、权重哈希与公共预热**。短测初始化占比高，所以n16实际rollout达到89.75%，整段仍只有30.49%；不能把89.75%当全程或连续3小时保证。此前旧入口全50task/100完整回合的72.04%整段、82.16%活跃窗口是另一场已归档测试，也不能当本次新适配器完整2500回合的结果。
