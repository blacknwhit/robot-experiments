# Eval-only r3 overlay for the original SCO experiment

The r2 training payload, resolved recipes, complete checkpoints and GPU
save/restore acceptance are immutable. Deploy this overlay separately at
`starVLA_pi05_embedding_position_v3_debug_20261020/eval_repair_r3_20261022`.
Copy the current `scripts/evaluate.py` into that directory, generate a SHA256
manifest for the eight runtime files, and verify both manifests before running.
Do not regenerate or replace the original r2 release manifest to make an edit
appear compatible with trained checkpoints.

The simulator receives its own explicit libffi preload on this SCO image.
Training and policy processes keep their pinned NCCL environment. The portable
evaluator clears inherited `LD_PRELOAD` for simulation and offers
`--sim-ld-preload` for deployments requiring a pinned ABI library. Other
clusters must validate their own simulator environment instead of copying
this machine-specific path. CUDA policy affinity and EGL renderer affinity
remain separate. Faulthandler, actual EGL UUID/image checks, RPC/step counters
and full-episode trace validation cover runtime failures beyond import tests.

The original A1500 SIGABRT did not reproduce in a complete, unchanged r2
two-episode run on fslgpu. Therefore libffi is **not a confirmed root cause**.
This overlay fixes a known environment-isolation defect and adds diagnostics;
acceptance requires complete GPU episodes from the final combination, followed
by an ACP-worker replay. Do not attribute a single transient native abort to
the model, loss or a specific library without additional evidence.

The controller skips A short training only after deep validation of the latest
complete compatible checkpoint at or beyond 1500, then performs A evaluation
on the actual worker before starting B's authorized 1500-step short run. A
single diagnostic replay of the unmodified r2 evaluator is recorded first;
its failure is preserved independently of the new evaluator's required gate. It
does not start a 70k experiment. Both eval outputs live in `home/evaluation_r3`.
Known application errors still write `HOLD_FOR_MAIN.json`; do not automatically
delete the hold or introduce a second recovery owner.

2026-10-22 validation: all 22 local tests pass. On fslgpu H100, the final
overlay completed two full episodes, 1800 environment steps and 360 policy
calls. The actual EGL UUID matched the H100; three 256x256 RGB views were
valid. Both episode trace hashes and raw Boolean outcomes matched the
unmodified entry exactly. This establishes the bounded evaluator contract,
not a benchmark result or the root cause of the old SIGABRT. The immutable
965-file training payload and the separate 8-file eval manifest were verified.
The resumed 2-H100 ACP job is `pt-gnvm1t6r`, native monitor
`98e047be-fde0-4e28-b6ac-6582308b7005` at ten-minute intervals. Its first
worker was Pending at 20:33 Beijing; ACP replay and B1500 remain outstanding.
