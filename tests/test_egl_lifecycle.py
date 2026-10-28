import ctypes
import hashlib
import inspect
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from egl_lifecycle import address, context_free, renderer_free, rewrite
import egl_lifecycle


class FakeEGL:
    EGL_NO_SURFACE = None
    EGL_NO_CONTEXT = None
    EGL_DRAW = 1
    EGL_READ = 2

    def __init__(self):
        self.current = ctypes.c_void_p(2)
        self.events = []

    def eglGetCurrentContext(self): return self.current
    def eglGetCurrentDisplay(self): return 'display'
    def eglGetCurrentSurface(self, kind): return kind
    def eglQueryAPI(self): return 12450
    def eglBindAPI(self, api): self.events.append(('api', api)); return True
    def eglMakeCurrent(self, display, draw, read, context):
        self.current = context
        self.events.append(('bind', address(context)))
        return True

    def eglDestroyContext(self, display, context):
        self.events.append(('destroy', address(context), address(self.current)))
        return True


def original_read(self):
    value = self.allocate()
    return self.native_read(value)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        egl_lifecycle._ERRORS.clear()
        egl_lifecycle._QUARANTINE.clear()

    def setup_renderer(self):
        egl = FakeEGL()
        module = SimpleNamespace(EGL=egl, EGL_DISPLAY='display')
        owner = SimpleNamespace(_context=ctypes.c_void_p(1))
        owner.make_current = lambda: egl.eglMakeCurrent('display', None, None, owner._context)
        con = SimpleNamespace(free=lambda: egl.events.append(('free-con', address(egl.current))))
        return egl, module, SimpleNamespace(gl_ctx=owner, con=con)

    def test_free_old_context_keeps_new_context_and_never_releases_thread(self):
        egl, module, renderer = self.setup_renderer()
        context_free(renderer.gl_ctx, module)
        self.assertEqual(address(egl.current), 2)
        self.assertEqual(egl.events, [('destroy', 1, 2)])
        context_free(renderer.gl_ctx, module)
        self.assertEqual(len(egl.events), 1)

    def test_renderer_frees_under_owner_then_restores_new_context(self):
        egl, module, renderer = self.setup_renderer()
        renderer_free(renderer, module)
        self.assertIn(('free-con', 1), egl.events)
        self.assertIn(('destroy', 1, None), egl.events)
        self.assertEqual(address(egl.current), 2)
        self.assertFalse(hasattr(renderer, 'con'))
        before = list(egl.events)
        renderer_free(renderer, module)
        self.assertEqual(egl.events, before)

    def test_partial_initialization_cannot_free_other_context_objects(self):
        egl, module, renderer = self.setup_renderer()
        renderer.gl_ctx._context = None
        renderer_free(renderer, module)
        self.assertFalse(any(event[0] == 'free-con' for event in egl.events))
        self.assertEqual(address(egl.current), 2)
        self.assertIs(egl_lifecycle._QUARANTINE[-1], renderer)
        self.assertTrue(hasattr(renderer, 'con'))
        with self.assertRaises(RuntimeError): egl_lifecycle.assert_healthy()

    def test_failed_native_free_is_quarantined_before_restoring_other_context(self):
        egl, module, renderer = self.setup_renderer()
        def fail(): raise RuntimeError('native failure')
        renderer.con.free = fail
        renderer_free(renderer, module)
        self.assertEqual(address(egl.current), 2)
        self.assertTrue(hasattr(renderer, 'con'))
        self.assertEqual(address(renderer.gl_ctx._context), 1)
        with self.assertRaisesRegex(RuntimeError, 'native failure'): egl_lifecycle.assert_healthy()

    def test_bind_after_allocation_and_refuse_unknown_source(self):
        digest = hashlib.sha256(textwrap.dedent(inspect.getsource(original_read)).encode()).hexdigest()
        repaired = rewrite(original_read, digest, [('    return self.native_read(value)',
                            '    self.gl_ctx.make_current()\n    return self.native_read(value)')])
        egl, module, renderer = self.setup_renderer()
        def allocate():
            egl.current = None
            return 'pixel-buffer'
        renderer.allocate = allocate
        renderer.native_read = lambda value: (value, address(egl.current))
        self.assertEqual(repaired(renderer), ('pixel-buffer', 1))
        with self.assertRaises(ValueError): rewrite(original_read, '0'*64, [])

    def test_restore_failure_is_visible_outside_python_finalizer(self):
        for throws in (False, True):
            egl_lifecycle._ERRORS.clear()
            egl, module, renderer = self.setup_renderer()
            def bind(api):
                if throws: raise RuntimeError('EGL error')
                return False
            egl.eglBindAPI = bind
            renderer_free(renderer, module)
            with self.assertRaisesRegex(RuntimeError, 'EGL'): egl_lifecycle.assert_healthy()


if __name__ == '__main__': unittest.main()
