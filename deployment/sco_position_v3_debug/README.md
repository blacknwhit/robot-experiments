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
retained separately. The r1 manifest covers 964 deployed files.

Revision r2, 2026-10-22: the actual r1 worker verified the CUDA correction with
separate old/fixed subprocesses and passed a two-H100 NCCL all-reduce. It then
failed before the first training update because the real RoboCasa overlay
preceded the policy import shim during lazy model construction. The policy
environment consequently attempted to import MuJoCo. V3 policy paths now put
the shim first; simulator calls still exclude it and import the real simulator.
Historical variants retain their previous import order.

The exact lazy OpenPI configuration/model imports pass on SCO for both A and B;
the simulator also imports under evaluate.py's actual subprocess environment.
Local validation passes 21 repository tests and two collector regression tests.
The corrected smoke and short run names end in `_r2`; the old zero-update
recipe remains preserved. No learned state was discarded. The r2 remote
manifest covers 965 files. GPU training/save-restore, action inference and EGL
rendering remain pending. The collector can audit a terminal last-known job
after native monitor removal without claiming it has a live recovery owner.
