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
