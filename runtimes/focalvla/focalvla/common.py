import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))})


def lr_at(completed, config):
    """LR for the next optimizer update; scheduler state is the committed step."""
    import math
    n = completed + 1
    warmup = config['warmup_steps']
    if n <= warmup:
        return config['peak_lr'] * n / warmup
    phase = min(1., (n - warmup) / (config['total_steps'] - warmup))
    return config['final_lr'] + .5 * (config['peak_lr'] - config['final_lr']) * (1 + math.cos(math.pi * phase))

