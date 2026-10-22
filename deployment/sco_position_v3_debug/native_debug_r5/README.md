# Native rendering diagnostic, no training

The training pipeline stays on its r4 HOLD after SIGABRT in
`mujoco.mjr_readPixels`. This separate task reads the existing A1500 weight and
the immutable r2/r4 payloads. It never clears the training HOLD, updates weights,
changes loss/batch/data, or starts B1500/70k training.

GDB 9.2 and dependencies are extracted from Ubuntu packages under this directory
only. Package hashes are recorded in gdb_prepare.json; the debugger and source
files are covered by manifest.json. No system package installation is needed.
Debugger-only LD_LIBRARY_PATH/PYTHONHOME/PYTHONPATH are restored to the original
simulator values before the inferior starts. CPU preflight verifies normal
exit, deliberate child SIGABRT/native backtrace capture, and timeout cleanup.
These are debugger checks, not a reproduction of the MuJoCo fault.

The native ACP controller requires the same two-H100 allocation as the failing
worker, records CUDA/EGL UUID maps, and refuses to render if EGL index zero is
outside the allocated CUDA devices. It reserves at most three diagnostic
replays across native generations. A captured native signal stops the task
with a terminal diagnostic record; three passes mean only that the intermittent
abort did not reproduce. Timeouts and interrupts are not classified as the
original native crash. Any preparation failure also stops once for inspection.

Each replay uses the existing strict A1500 evaluator, seed7, two episodes,
900-step horizon, five-step replanning, and real EGL. Artifacts include the
native backtrace of all threads, registers, loaded libraries, simulator audit,
strict checkpoint handshake, and raw result/trace arrays when completed.
No simulator image/physics/protocol optimization is introduced here.

The task owns only control/diagnostic.lock and a shared read lock on the old
pipeline lock. Native SCO auto-retrain at ten minutes handles external task
interruptions; TERMINAL.json prevents an automatic diagnostic loop after a
captured failure or the bounded budget. The main training native owner remains
absent until a separate, verified repair is deployed by the main conversation.
