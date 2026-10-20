# Original SCO bounded A/B validation

These launch files are specific to the original SCO installation. Portable
training and evaluation remain in `scripts/`; do not run this controller on
another cluster without translating its explicit paths and preparing its
release manifest and resolved configurations.

Revision r1, 2026-10-21, preserves A/B model, objective, data, optimizer and
the 1500-step engineering limit. It adds the system NVIDIA driver library
directories to the launch environment. The old launch selected the image's
CUDA 12.6 compatibility library and failed with CUDA Error 803 on the real
ACP worker before the first training update.

On the CPU access container, separate-process `dladdr` checks resolve the old
path to `/usr/local/cuda-12.6/compat/libcuda.so.1` (driver API 12060) and the
corrected path to `/lib/x86_64-linux-gnu/libcuda.so.1` (API 13000). This checks
library selection, not GPU initialization. The controller records both
environments in independent GPU probe processes and only continues after the
corrected environment initializes two H100 devices and performs an allocation.

An application error writes `HOLD_FOR_MAIN.json` and exits without a blind
retry loop. Platform success is therefore not application acceptance.
The old task's full error record is preserved. Only after confirming its
worker had exited, the pipeline lock was free, and no training recipe or
checkpoint existed was r1 deployed and submitted.

Remote evidence and release manifests are stored in the experiment root's
`control/` and `repair_r1/`. The immutable r0 archive and its manifest are
retained separately. The r1 manifest covers 964 deployed files. Actual GPU,
save/restore, action-inference and EGL acceptance remain pending.
