"""Headless RGB adapter using MuJoCo's native Renderer and owned cleanup.

The model, data, cameras, scene options and MSAA stay with robosuite. Only its
OpenGL context / framebuffer wrapper is replaced, in this simulator process.
Depth, segmentation and interactive perturbations are deliberately unsupported.
"""
import ctypes
import os
import sys
import weakref

import numpy as np

VERSION = 'mujoco331_native_rgb_owned_v1'
_contexts = weakref.WeakSet()
_quarantine = []
_errors = []


def _address(value):
    return ctypes.cast(value, ctypes.c_void_p).value if value else None


def _release(self):
    renderer = getattr(self, '_native_rgb', None)
    if renderer is None:
        return
    from OpenGL import EGL
    previous = EGL.eglGetCurrentContext()
    display = EGL.eglGetCurrentDisplay()
    draw = EGL.eglGetCurrentSurface(EGL.EGL_DRAW)
    read = EGL.eglGetCurrentSurface(EGL.EGL_READ)
    api = EGL.eglQueryAPI()
    owner = renderer._gl_context
    restore = bool(previous) and _address(previous) != _address(owner._context)
    try:
        owner.make_current()
        # MuJoCo 3.3.1 Renderer.close() closes GL before MjrContext. Explicitly
        # free model GPU resources first, while their owning context is current.
        if renderer._mjr_context is not None:
            renderer._mjr_context.free()
            renderer._mjr_context = None
        renderer.close()
        self._native_rgb = None
        self.con = self.gl_ctx = None
        self._pixels = None
    except BaseException as exc:
        # Do not let an unfreed native context die under another live context.
        _quarantine.append(self)
        _errors.append(type(exc).__name__ + ': native renderer cleanup failed')
    finally:
        try:
            if not EGL.eglBindAPI(api):
                _errors.append('Could not restore EGL API')
            if restore and not EGL.eglMakeCurrent(display, draw, read, previous):
                _errors.append('Could not restore the previous EGL context')
        except BaseException as exc:
            _errors.append(type(exc).__name__ + ': EGL restore failed')


def _init(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
    import mujoco
    if not offscreen:
        raise ValueError('Native RGB adapter supports offscreen evaluation only')
    self._native_rgb = None
    self._pixels = None
    self.con = self.gl_ctx = None
    self.sim, self.model, self.data = sim, sim.model, sim.data
    self.offscreen, self.device_id = True, device_id
    self._size = None
    # Preserve the original constructor's forward/register ordering.
    sim.forward()
    sim.add_render_context(self)
    self.scn = mujoco.MjvScene(sim.model._model, maxgeom=10000)
    self.cam = mujoco.MjvCamera()
    self.cam.fixedcamid = 0
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    self.vopt = mujoco.MjvOption()
    self.pert = mujoco.MjvPerturb()
    self.pert.active = 0
    self.pert.select = 0
    self.pert.skinselect = -1
    _contexts.add(self)


def _render(self, width, height, camera_id=None, segmentation=False):
    import mujoco
    if segmentation or self.pert.active or self.pert.active2:
        raise ValueError('This evaluation adapter supports RGB without interactive perturbations')
    if self._native_rgb is None or self._size != (width, height):
        _release(self)
        assert_healthy()
        renderer = mujoco.Renderer(self.model._model, height=height, width=width, max_geom=10000)
        # Keep externally configured scene flags/stereo on the SAME scene object.
        renderer._scene = self.scn
        self._native_rgb = renderer
        self._size = (width, height)
        self.con, self.gl_ctx = renderer._mjr_context, renderer._gl_context
        if (type(self.gl_ctx).__module__ != 'mujoco.egl'
                or type(self.gl_ctx).__name__ != 'GLContext'
                or not self.gl_ctx._context):
            _release(self)
            raise RuntimeError('MuJoCo was imported with a non-EGL backend; start a fresh simulator process')
    if camera_id is not None:
        self.cam.type = (mujoco.mjtCamera.mjCAMERA_FREE if camera_id == -1
                         else mujoco.mjtCamera.mjCAMERA_FIXED)
        self.cam.fixedcamid = camera_id
    self._native_rgb.update_scene(self.data._data, camera=self.cam, scene_option=self.vopt)
    # Zero initialization makes a failed/unwritten read observable, rather than
    # silently returning an earlier camera frame from np.empty's recycled memory.
    out = np.zeros((height, width, 3), dtype=np.uint8)
    top_up = self._native_rgb.render(out=out)
    # robosuite expects bottom-up; its normal Gym wrapper performs the final flip.
    self._pixels = top_up[::-1].copy()


def _read(self, width, height, depth=False, segmentation=False):
    if depth or segmentation:
        raise ValueError('Depth and segmentation are not part of this RGB evaluation adapter')
    if self._pixels is None or self._size != (width, height):
        raise RuntimeError('Pixel read without a matching rendered frame')
    return self._pixels.copy()


def _resize(self, width, height):
    _release(self)
    assert_healthy()


def _delete(self):
    if not sys.is_finalizing():
        _release(self)


def assert_healthy():
    if _errors:
        raise RuntimeError('Native RGB resource error: ' + '; '.join(_errors))


def retain_environment(env):
    live = env.unwrapped.env.sim._render_context_offscreen
    if getattr(type(live), '_native_rgb_version', None) != VERSION:
        raise RuntimeError('Environment did not use the native RGB adapter')
    for context in list(_contexts):
        if context is not live:
            _release(context)
    assert_healthy()
    return {'adapter': VERSION, 'engine': 'mujoco', 'backend': 'egl',
            'msaa': int(live.con.offSamples), 'max_geom': int(live.scn.maxgeom),
            'gl_context_type': type(live.gl_ctx).__module__ + '.' + type(live.gl_ctx).__name__}


def release_all():
    for context in list(_contexts):
        _release(context)
    assert_healthy()


def install():
    if os.environ.get('MUJOCO_GL') != 'egl' or os.environ.get('PYOPENGL_PLATFORM') != 'egl':
        raise RuntimeError('Set EGL environment variables before starting the simulator Python')
    import mujoco
    from robosuite.utils.binding_utils import MjRenderContext
    if mujoco.__version__ != '3.3.1':
        raise RuntimeError('The native RGB adapter is validated with mujoco==3.3.1')
    if getattr(MjRenderContext, '_native_rgb_version', None) == VERSION:
        return VERSION
    MjRenderContext.__init__ = _init
    MjRenderContext.render = _render
    MjRenderContext.read_pixels = _read
    MjRenderContext.update_offscreen_size = _resize
    MjRenderContext.__del__ = _delete
    MjRenderContext._native_rgb_version = VERSION
    return VERSION
