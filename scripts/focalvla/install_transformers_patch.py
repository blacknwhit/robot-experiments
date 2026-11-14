"""Install the vendored OpenPI patch in the active POLICY virtual environment."""
from pathlib import Path
import shutil
import importlib.metadata
import transformers

assert importlib.metadata.version('transformers') == '4.53.2'
source = Path(__file__).resolve().parents[2]/'runtimes/focalvla/vendor/openpi/models_pytorch/transformers_replace'
destination = Path(transformers.__file__).parent
for file in source.rglob('*.py'):
    target = destination/file.relative_to(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(file,target)
print(f'Installed {len(list(source.rglob("*.py")))} OpenPI compatibility files into {destination}')
