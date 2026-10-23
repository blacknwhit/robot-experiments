# EXP-003：3D轨迹v2（Query / Embedding对照）

[返回实验总览](../../EXPERIMENTS.md)

## 实验定义

这一次实验包含两组，以机械臂＋直接物体3D轨迹作辅助监督。共享源码`runtimes/trajectory_v2/`，与本轮EXP-004的完整位置监督及时间点头分开。

| 组 | 配置 | Z_t来源 | 历史70k结果 |
|---|---|---|---|
| Query3D-v2 | [query3d_v2.yaml](../../configs/query3d_v2.yaml) | 插入50个Query | 373/2500，14.92% |
| Embedding3D-v2 | [embedding3d_v2.yaml](../../configs/embedding3d_v2.yaml) | 原动作prefix＋独立训练期Embedding | 443/2500，17.72% |

旧目标为相邻位移＋累计轨迹项；不能将其输出语义或同形状checkpoint冒充EXP-004的完整起点相对位置＋相邻差分。两组均没有超过EXP-002这里记录的559/2500，但这些数字来自历史协议。

## 资源与执行

- 两组均从原始`pi05_base`独立初始化，运行/权重/评测输出互相分离。
- 机械臂/物体1024点原始GT可供EXP-004复用，前提是来源、范围、坐标和配置身份一致；v3改变reader/监督，不要求为每组重造几何。
- 本实验的历史完整配置与操作见[公共历史交接](../../HANDOFF.zh-CN.md)，选择`query3d_v2`或`embedding3d_v2`，不要用v3 trainer或配置。
- 新场景重测使用[公共固定场景评测](../../docs/ASTRA_FIXED_SCENES.zh-CN.md)，结果另存，历史373/443分不改写。

这次文档归档没有重新执行70k训练或评测，也不搬动已有GT、模型或完整训练状态。
