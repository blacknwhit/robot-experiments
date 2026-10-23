"""Exercise the lazy policy construction imports without allocating a model."""
import pyarrow  # Match the actual training entry import order.
import argparse
import importlib
import json
from pathlib import Path
import sys
import yaml
from common import activate, POSITION_VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    variant = cfg['portable']['variant']
    if variant not in POSITION_VARIANTS:
        raise ValueError('This bounded preflight is for the V3 engineering variants')
    runtime = activate(variant)
    # Match the order that previously failed inside OpenPIPi05.__init__.
    importlib.import_module('starVLA.training.train_starvla')
    importlib.import_module('starVLA.training.position_point_trainer')
    loaded = {}
    for name in ('openpi.training.config', 'openpi.models_pytorch.pi0_pytorch',
                 'openpi.shared.normalize', 'openpi.transforms', 'openpi.models.model'):
        module = importlib.import_module(name)
        loaded[name] = module.__file__
    package = importlib.import_module('robocasa')
    expected = runtime / 'examples/pi05_robocasa365/server_compat/robocasa/__init__.py'
    assert Path(package.__file__).resolve() == expected.resolve(), package.__file__
    assert not any(n == 'mujoco' or n.startswith('mujoco.') for n in sys.modules)
    config = importlib.import_module('openpi.training.config').get_config(
        cfg['framework']['pi05']['openpi_config'])
    assert config.model is not None
    print('POLICY_IMPORT_PREFLIGHT ' + json.dumps(dict(passed=True, variant=variant,
          robocasa=package.__file__, lazy_modules=loaded, model_allocated=False,
          gpu_acceptance=False)), flush=True)


if __name__ == '__main__':
    main()
