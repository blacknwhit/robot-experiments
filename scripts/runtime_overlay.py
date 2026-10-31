"""Build a separate Transformers 4.53.2 overlay for each source runtime."""
import argparse, importlib.metadata, importlib.util, shutil, tempfile
from common import VARIANTS, runtime, sha256

def build(variant):
    if importlib.metadata.version('transformers')!='4.53.2':raise RuntimeError('Install transformers==4.53.2 first')
    root=runtime(variant);target=root/'runtime';replacement=root/'playground/Code/openpi_robocasa365/src/openpi/models_pytorch/transformers_replace'
    if not target.exists():
        source=__import__('pathlib').Path(importlib.util.find_spec('transformers').origin).parent
        with tempfile.TemporaryDirectory(prefix='.overlay-',dir=root) as temp:
            stage=__import__('pathlib').Path(temp)/'runtime'
            shutil.copytree(source,stage/'transformers',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copytree(replacement,stage/'transformers',dirs_exist_ok=True)
            stage.rename(target)
    for src in replacement.rglob('*.py'):
        dst=target/'transformers'/src.relative_to(replacement)
        if not dst.is_file() or sha256(src)!=sha256(dst):raise RuntimeError(f'Overlay differs: {dst}')
    print('OVERLAY_OK',variant,target)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    for v in ([a.variant] if a.variant else ['baseline','query_fixed','query3d_v2','embedding3d_pos_a','pri4r_paper']):build(v)
