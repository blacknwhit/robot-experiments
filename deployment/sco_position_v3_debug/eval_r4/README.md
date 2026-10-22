# r4: sequential evaluator TCP port preflight

r3's ACP worker passed actual two-H100 CUDA/NCCL and the unchanged A1500
evaluator completed two full episodes. The next evaluator failed before
starting its policy server with EADDRINUSE in the plain socket bind probe.
B1500 was not started. The old server had exited; the second check followed
only about 37 seconds later. The same plain bind error was reproduced on SCO
after an actively closed connection entered TIME_WAIT. The fixed check uses
SO_REUSEADDR on POSIX (matching asyncio servers), never SO_REUSEPORT, and keeps
the active-listener rejection. Windows uses SO_EXCLUSIVEADDRUSE.

All six launch-environment tests pass on Linux, including two immediate
same-port restart cycles, the old bind's expected EADDRINUSE, and rejection
of a real listening server. Windows passes five tests and skips the
Linux-specific TIME_WAIT test. GPU validation injects a fresh TIME_WAIT
connection immediately before the real evaluator's port preflight, then runs
the unchanged A1500 policy and full two-episode simulator workload.

The previous SIGABRT has not reproduced on either fslgpu or this ACP worker;
its root cause remains unconfirmed. This port failure is a separately
identified orchestration defect. No model, training recipe, weight, simulator
task/seed/horizon, or success criterion is changed.

Runtime overlay: eval_repair_r4_20261022, separate eight-file manifest. The
965-file r2 training release remains byte-identical. The r4 controller reuses
the completed r3 original-entry diagnostic receipt instead of rerunning it,
deep-validates A1500, requires the repaired A evaluation, and then continues
the already-authorized B1500 short experiment. It never launches a 70k run.
