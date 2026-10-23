# B1500短训与评测分开执行（r8）

r8只继续已授权的B工程短训，正式训练未启动。训练配置保持2H100/B32/acc1/global64、
seed42、G8、原模型/GT/位置与差分loss/Adam/LR不变；scheduler仍按原100k设计，
stop_after_steps严格为1500。不会把B smoke3工程权重加载为短训初始权重。

入口要求r7当前源代码SHA和真实A1500/Bsmoke3 EGL验收通过。原ACP评测HOLD保留，
不清除、不伪称修复；r8使用独立control并独占原pipeline锁及run-local writer锁，
只启动B。已有完整兼容checkpoint则恢复模型/Adam/scheduler/RNG/data clocks；
缺少PORTABLE_COMPLETE或不完整状态跳过，已有base claim却无完整状态则HOLD，
禁止静默重复从base。首次更新额外完整保存，之后1000及最终1500保存。

主进程完成时深验1500模型、两rank Adam/RNG、scheduler/data clocks、轨迹契约、
step1/1000/1500保存marker和有限loss，写B_SHORT1500_COMPLETE.json并退出。
marker明确evaluation_pending=true，平台SUCCEEDED不代表全部训练后评测完成。

本地5项恢复保护测试通过；最终payload在fslgpu以CPU方式通过实际训练入口参数、
延迟导入及不创建模型的检查。GPU更新使用原已验收训练代码，新的ACP worker仍需
实际2H100/CUDA/NCCL门槛，不能把CPU预检当作该任务GPU验收。

标准实际成员GPU配额0拒绝后提交2H100 SPOT：pt-z4luzdfk。
native monitor：8b64a2c5-02b5-474d-ac18-5aeb27b5c40d，autoRetrain=true、10min。
17:11核验worker Pending；以后以monitor currentJobName为准，不重复提交。
配套r9为一次有界依赖评测，不提交或恢复训练。
