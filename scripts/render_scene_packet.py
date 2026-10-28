"""Lossless MuJoCo 3.3.1 abstract-scene IPC; no physics replay or GL in sender.

Only pointer-free mjvGeom/Light/GLCamera records use bulk copies. Their layouts
are checked against every exposed array before use. All remaining public scene
fields are copied explicitly; a new/unsupported field fails closed.
"""
import ctypes as C
import sys
import numpy as np

I, F, B = C.c_int, C.c_float, C.c_ubyte


class Geom(C.Structure):
    _fields_ = [(n, I) for n in ('type', 'dataid', 'objtype', 'objid', 'category', 'matid', 'texcoord', 'segid')] + [
        ('size', F*3), ('pos', F*3), ('mat', F*9), ('rgba', F*4),
        ('emission', F), ('specular', F), ('shininess', F), ('reflectance', F),
        ('label', C.c_char*100), ('camdist', F), ('modelrbound', F), ('transparent', B)]


class Light(C.Structure):
    _fields_ = [('pos', F*3), ('dir', F*3), ('attenuation', F*3), ('cutoff', F), ('exponent', F),
                ('ambient', F*3), ('diffuse', F*3), ('specular', F*3), ('headlight', B),
                ('directional', B), ('castshadow', B), ('bulbradius', F)]


class Camera(C.Structure):
    _fields_ = [('pos', F*3), ('forward', F*3), ('up', F*3)] + [
        (n, F) for n in ('frustum_center', 'frustum_width', 'frustum_bottom', 'frustum_top',
                        'frustum_near', 'frustum_far')] + [('orthographic', I)]


SCALARS = ('ngeom', 'nlight', 'nflex', 'nskin', 'enabletransform', 'scale', 'stereo',
           'framewidth', 'flexedgeopt', 'flexfaceopt', 'flexskinopt', 'flexvertopt')
ARRAYS = ('flags', 'translate', 'rotate', 'framergb', 'flexedge', 'flexedgeadr', 'flexedgenum',
          'flexface', 'flexfaceadr', 'flexfacenum', 'flexfaceused', 'flexnormal', 'flextexcoord',
          'flexvert', 'flexvertadr', 'flexvertnum', 'skinfacenum', 'skinnormal', 'skinvert',
          'skinvertadr', 'skinvertnum')
FIELDS = set(SCALARS + ARRAYS + ('maxgeom', 'geomorder', 'geoms', 'lights', 'camera'))


def checked_address(objects, layout, anchor):
    first = objects[0]
    address = getattr(first, anchor).ctypes.data - getattr(layout, anchor).offset
    for index in (0, 1):
        if index >= len(objects):
            continue
        obj = objects[index]
        for name, field_type in layout._fields_:
            if name == 'label':
                continue  # Unused native geom slots contain uninitialized char data.
            value = getattr(obj, name)
            if isinstance(value, np.ndarray):
                expected = address + index*C.sizeof(layout) + getattr(layout, name).offset
                if value.ctypes.data != expected or value.nbytes != C.sizeof(field_type):
                    raise RuntimeError('Unsupported native scene layout: ' + name)
    return address


class ScenePacket:
    def __init__(self, scene):
        import mujoco
        if mujoco.__version__ != '3.3.1' or sys.byteorder != 'little' or C.sizeof(I) != 4:
            raise RuntimeError('Scene packets require the pinned MuJoCo 3.3.1 ABI')
        actual = {name for name in dir(scene) if not name.startswith('_')}
        if actual != FIELDS:
            raise RuntimeError('Unsupported scene fields: ' + repr(actual ^ FIELDS))
        self.scene = scene
        self.pods = [('geoms', Geom, checked_address(scene.geoms, Geom, 'size')),
                     ('lights', Light, checked_address(scene.lights, Light, 'pos')),
                     ('camera', Camera, checked_address(scene.camera, Camera, 'pos'))]

    def capture(self):
        s = self.scene
        sizes = {'geoms': s.ngeom, 'lights': s.nlight, 'camera': 2}
        return {'scalars': {name: getattr(s, name) for name in SCALARS},
                'arrays': {name: getattr(s, name).copy() for name in ARRAYS},
                'geomorder': s.geomorder[:s.ngeom].copy(),
                'pods': {name: C.string_at(address, sizes[name]*C.sizeof(layout))
                         for name, layout, address in self.pods}}

    def restore(self, packet):
        s = self.scene
        scalars = packet['scalars']
        if set(scalars) != set(SCALARS) or set(packet['arrays']) != set(ARRAYS):
            raise ValueError('Incomplete scene packet')
        if not 0 <= scalars['ngeom'] <= s.maxgeom or not 0 <= scalars['nlight'] <= len(s.lights):
            raise ValueError('Scene packet exceeds allocated capacity')
        if scalars['nflex'] != s.nflex or scalars['nskin'] != s.nskin:
            raise ValueError('Scene packet belongs to a different model')
        sizes = {'geoms': scalars['ngeom'], 'lights': scalars['nlight'], 'camera': 2}
        for name, layout, address in self.pods:
            data = packet['pods'][name]
            if len(data) != sizes[name]*C.sizeof(layout):
                raise ValueError('Invalid scene record length: ' + name)
            C.memmove(address, data, len(data))
        for name, value in scalars.items():
            if name not in ('nflex', 'nskin'):
                setattr(s, name, value)
        for name, value in packet['arrays'].items():
            target = getattr(s, name)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError('Scene array identity mismatch: ' + name)
            target[:] = value
        order = packet['geomorder']
        if order.shape != (s.ngeom,) or order.dtype != s.geomorder.dtype:
            raise ValueError('Invalid geom order')
        s.geomorder[:s.ngeom] = order
