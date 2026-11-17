# FOCAL 离线缓存辅助提取

在原5个双卡准备任务保持运行时，可各增一个双卡助手。原任务正向处理，助手按同一演示分片反向处理。助手每集复查原目录完成标记，完整原结果直接跳过；在途工作和新标签写独立辅助目录，不与原进程并发写同一集。

助手使用相同资源文件、计划、教师权重、三视角、11帧、518/630分辨率和B8特征提取实现。辅助prepared目录的PLAN.json、CACHE_IDENTITY.json、episodes.json、records.npy、repeats.npy、draws.npy应只读链接到原prepared；annotations和cache必须为独立目录。

```bash
export PYTHONPATH="$PWD/runtimes/focalvla${PYTHONPATH:+:$PYTHONPATH}"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/focalvla/prepare_assist.py --resources "$RESOURCES" \
  --primary "$PREPARED" --prepared "$ASSIST_PREPARED" \
  --status-dir "$ASSIST_STATUS" --shard-index 0
```

五份助手的shard-index分别为0到4。保持原任务同样的5×2分片，每片6000演示。无需重启原实例来加载新分片规则。跨进程恢复仍由作业平台负责。

收尾需要单独的所有权交接：确认该片6000集在主/辅助目录的完整结果并集中全部覆盖，停止对应原写入者并确认进程退出，才可调用assist_cache.merge。合并优先保留原完整缓存，原半成品移入归档，辅助目录补足缺失结果。标签不一致、身份变化、缺文件、帧索引不符会拒绝合并；不能仅依据完成计数覆盖正在写的文件。

全部30000集/4076006目标合并并校验后，统一执行原focalvla.prepare seal。到READY.json为止，本流程不启动VLA训练，也不生成全量PNG或images.bin。原GB64/70k训练配置保留供后续使用。

```bash
python scripts/focalvla/test_assist_cache.py -v
```

测试覆盖主结果优先、辅助结果合并、半成品保留、合并中断后重入、身份与标签冲突拒绝、原进程新完成结果的动态识别。教师网络和训练运行时代码保持原样；新增代码仅控制数据处理顺序、隔离输出和合并。
