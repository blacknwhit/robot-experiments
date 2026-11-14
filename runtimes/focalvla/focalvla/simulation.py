"""Restore recorded simulator state; privileged information stays offline."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rewrite_assets(xml_bytes, roots):
    xml = ET.fromstring(xml_bytes)
    assets = []
    for element in xml.iter():
        old = element.get('file')
        if not old:
            continue
        new = None
        for package, root in roots.items():
            marker = f'/{package}/models/assets/'
            if marker in old:
                new = Path(root) / old.split(marker, 1)[1]
                break
        if new is None:
            raise ValueError(f'Unmapped asset {old}; provide its package asset root.')
        if not new.is_file():
            raise FileNotFoundError(new)
        element.set('file', str(new.resolve()))
        assets.append(str(new))
    for compiler in xml.findall('compiler'):
        compiler.attrib.pop('meshdir', None)
        compiler.attrib.pop('texturedir', None)
    return ET.tostring(xml, encoding='unicode'), assets


def restore_state(model, data, flat):
    import mujoco
    flat = np.asarray(flat, dtype=np.float64)
    expected = 1 + model.nq + model.nv + model.na
    if flat.shape != (expected,) or not np.isfinite(flat).all():
        raise ValueError(f'Invalid recorded state: {flat.shape}; expected {(expected,)}')
    data.time = flat[0]
    data.qpos[:] = flat[1:1 + model.nq]
    data.qvel[:] = flat[1 + model.nq:1 + model.nq + model.nv]
    if model.na:
        data.act[:] = flat[1 + model.nq + model.nv:]
    mujoco.mj_kinematics(model, data)

