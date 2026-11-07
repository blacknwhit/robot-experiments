"""Server-only RoboCasa import shim.

OpenPI's inference config imports RoboCasa training-dataset symbols at module
load time even though policy inference never instantiates those datasets. The
real simulator package is intentionally not imported here because it pins a
different NumPy ABI. Simulator processes never include this shim on PYTHONPATH.
"""

