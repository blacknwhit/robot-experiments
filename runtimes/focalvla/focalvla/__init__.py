"""FOCAL-VLA on the three-view RoboCasa365 PI0.5 adapter."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

