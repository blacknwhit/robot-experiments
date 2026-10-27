"""Process-local EGL lifetime repair for the pinned robosuite renderer.

No camera, MSAA, framebuffer format or rendering algorithm is changed. An old
renderer may be collected while a new renderer is active; free its GL objects
under its own context and restore the new context afterward.
"""
import ctypes
import hashlib
import inspect
import textwrap

VERSION = 'owned_egl_context_episode_gc_v1'
_QUARANTINE = []
_ERRORS = []
SOURCES = {
    'egl.free': 'fb789e1a9a04d71d6714bc01025cb561516b1f7d30de562bf1746b72cf0e9d83',
    'renderer.__del__': '3aa4f8cd85d8bd7c12c363d4e9c0035481d79382cb3a21e16f37bff6cd927762',
    'renderer.read_pixels': '8134a180223b0303fa97df49fcfa889c5849cc3aee0e436e4282b9109b9e64a4',
    'renderer.render': '64066a316f725822c5887b056e4c870bd0c0bc19814ed450ed2edd6e5d555e0e',
    'renderer._set_mujoco_context_and_buffers': 'ea72a1c368b4597a090745f30f9f02b0dc0ec54a79a74b0ca1ea89e4c1e18c57',
    'renderer.update_offscreen_size': '24211f41ab5755a2b34866ca9ca39fcebfd72b79865883179353bc159e71e258',
}


def address(handle):
    return ctypes.cast(handle, ctypes.c_void_p).value if handle else None


def context_free(owner, module):
    egl = module.EGL
    context = getattr(owner, '_context', None)
    if not context:
        return
    if address(egl.eglGetCurrentContext()) == address(context):
        if not egl.eglMakeCurrent(module.EGL_DISPLAY, egl.EGL_NO_SURFACE,
                                  egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT):
            raise RuntimeError('Could not detach owned EGL context')
    if not egl.eglDestroyContext(module.EGL_DISPLAY, context):
        raise RuntimeError('Could not destroy owned EGL context')
    owner._context = None
    # eglReleaseThread would ALSO detach a different, still-live renderer.


def renderer_free(renderer, module):
    if getattr(renderer, '_owned_egl_released', False) or getattr(renderer, '_owned_egl_quarantined', False):
        return
    egl = module.EGL
    owner = getattr(renderer, 'gl_ctx', None)
    owned = getattr(owner, '_context', None)
    previous = egl.eglGetCurrentContext()
    display = egl.eglGetCurrentDisplay()
    draw = egl.eglGetCurrentSurface(egl.EGL_DRAW)
    read = egl.eglGetCurrentSurface(egl.EGL_READ)
    api = egl.eglQueryAPI()
    restore = bool(previous) and address(previous) != address(owned)
    released = False
    try:
        # Partial initialization must not free GL objects in somebody else's
        # context. Do not take _MjSim_render_lock: GC may run under that lock.
        if owned:
            owner.make_current()
            con = getattr(renderer, 'con', None)
            if con is not None:
                con.free()
            context_free(owner, module)
        elif getattr(renderer, 'con', None) is not None:
            raise RuntimeError('Native rendering resources have no live owning EGL context')
        released = True
    except BaseException as exc:
        # MjrContext's native destructor also frees GL objects. Keep the whole
        # owner alive on failure instead of dropping con in the restored context.
        renderer._owned_egl_quarantined = True
        _QUARANTINE.append(renderer)
        _ERRORS.append(repr(exc))
    finally:
        try:
            if not egl.eglBindAPI(api):
                _ERRORS.append('Could not restore EGL client API after cleanup')
            if restore and not egl.eglMakeCurrent(display, draw, read, previous):
                _ERRORS.append('Could not restore live EGL context after cleanup')
        except BaseException as exc:
            _ERRORS.append('EGL context restoration failed: ' + repr(exc))
        if released:
            renderer._owned_egl_released = True
            for name in ('con', 'gl_ctx', 'scn', 'cam', 'vopt', 'pert'):
                if hasattr(renderer, name):
                    delattr(renderer, name)


def assert_healthy():
    if _ERRORS:
        raise RuntimeError('EGL lifecycle cleanup failed: ' + '; '.join(_ERRORS))


def rewrite(original, expected, replacements):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != expected or original.__closure__ is not None:
        raise ValueError('Unsupported robosuite source; refusing an unverified EGL patch')
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Ambiguous renderer patch site')
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, original.__code__.co_filename + '#owned-egl', 'exec'), namespace)
    result = namespace[original.__name__]
    result.__qualname__ = original.__qualname__
    return result


def install():
    from robosuite.renderers.context import egl_context as module
    from robosuite.utils.binding_utils import MjRenderContext
    if getattr(MjRenderContext, '_owned_egl_version', None) == VERSION:
        return VERSION
    originals = {'egl.free': module.EGLGLContext.free,
                 **{'renderer.' + name: getattr(MjRenderContext, name)
                    for name in ('__del__', 'read_pixels', 'render',
                                 '_set_mujoco_context_and_buffers', 'update_offscreen_size')}}
    # Validate ALL methods before mutating any class.
    for name, original in originals.items():
        digest = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest()
        if digest != SOURCES[name]:
            raise ValueError('Unsupported robosuite EGL method: ' + name)
    sites = {
        'read_pixels': [('    mujoco.mjr_readPixels(', '    self.gl_ctx.make_current()\n    mujoco.mjr_readPixels(')],
        'render': [('    mujoco.mjr_render(', '    self.gl_ctx.make_current()\n    mujoco.mjr_render(')],
        '_set_mujoco_context_and_buffers': [
            ('    self.con = mujoco.MjrContext(', '    self.gl_ctx.make_current()\n    self.con = mujoco.MjrContext('),
            ('    mujoco.mjr_setBuffer(', '    self.gl_ctx.make_current()\n    mujoco.mjr_setBuffer(')],
        'update_offscreen_size': [('        self.con.free()', '        self.gl_ctx.make_current()\n        self.con.free()')],
    }
    repaired = {name: rewrite(originals['renderer.' + name], SOURCES['renderer.' + name], changes)
                for name, changes in sites.items()}
    module.EGLGLContext.free = lambda self: context_free(self, module)
    MjRenderContext.__del__ = lambda self: renderer_free(self, module)
    for name, method in repaired.items():
        setattr(MjRenderContext, name, method)
    MjRenderContext._owned_egl_version = VERSION
    return VERSION
