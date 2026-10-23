"""Verify the actual source bytes before fresh training or state restoration."""
from pathlib import Path
from common import ROOT, sha256


def verify_release(config, root=ROOT):
    root = Path(root).resolve()
    manifest = root / 'provenance/source_files.sha256'
    if config.get('trajectory_release_sha256') != sha256(manifest):
        raise ValueError('Training recipe belongs to another code release; use its frozen checkout')
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source manifest path escapes the release')
        if sha256(path) != expected:
            raise ValueError('Training source changed after sealing: ' + relative)
        checked += 1
    if not checked:
        raise ValueError('Empty source manifest')
    return checked
