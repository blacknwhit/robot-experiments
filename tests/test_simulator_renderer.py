import ctypes
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import simulator_renderer as adapter


class NativeRendererTests(unittest.TestCase):
    def setUp(self):
        adapter._errors.clear()
        adapter._quarantine.clear()

    def test_camera_scene_options_and_bottom_up_independent_pixels(self):
        updates=[]
        rgb=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        GLContext=type('GLContext',(),{'__module__':'mujoco.egl','_context':ctypes.c_void_p(1)})
        class Renderer:
            def __init__(self, model, **kwargs):
                self._mjr_context=object();self._gl_context=GLContext()
                self.model=model;self.kwargs=kwargs
            def update_scene(self,data,**kwargs):updates.append((data,kwargs))
            def render(self,out):
                self.zero_initialized=not out.any();out[:]=rgb;return out
        fake=NS(Renderer=Renderer,mjtCamera=NS(mjCAMERA_FREE=0,mjCAMERA_FIXED=2))
        context=NS(_native_rgb=None,_size=None,model=NS(_model=object()),data=NS(_data=object()),
                   scn=object(),cam=NS(type=2,fixedcamid=0),vopt=object(),pert=NS(active=0,active2=0))
        with patch.dict(sys.modules,{'mujoco':fake}):
            adapter._render(context,3,2,7)
            renderer=context._native_rgb
            self.assertTrue(renderer.zero_initialized)
            self.assertIs(renderer.model,context.model._model)
            self.assertIs(renderer._scene,context.scn)
            self.assertIs(updates[0][0],context.data._data)
            self.assertIs(updates[0][1]['scene_option'],context.vopt)
            self.assertIs(updates[0][1]['camera'],context.cam)
            self.assertEqual(context.cam.fixedcamid,7)
            np.testing.assert_array_equal(adapter._read(context,3,2),rgb[::-1])
            returned=adapter._read(context,3,2);returned[:]=0
            np.testing.assert_array_equal(adapter._read(context,3,2),rgb[::-1])
            adapter._render(context,3,2,None)
            self.assertEqual(context.cam.fixedcamid,7)
            adapter._render(context,3,2,-1)
            self.assertEqual(context.cam.type,0)
            self.assertIs(renderer,context._native_rgb)
            for kwargs in ({'depth':True},{'segmentation':True}):
                with self.assertRaises(ValueError):adapter._read(context,3,2,**kwargs)
            with self.assertRaises(ValueError):adapter._render(context,3,2,segmentation=True)
            context.pert.active2=1
            with self.assertRaises(ValueError):adapter._render(context,3,2)

    def release_fixture(self, fail=False):
        events=[];current=[ctypes.c_void_p(2)]
        def bind(display,draw,read,context):
            current[0]=context;events.append(('bind',adapter._address(context)));return True
        egl=NS(EGL_DRAW=1,EGL_READ=2,eglGetCurrentContext=lambda:current[0],
               eglGetCurrentDisplay=lambda:'display',eglGetCurrentSurface=lambda kind:kind,
               eglQueryAPI=lambda:12450,eglBindAPI=lambda api:True,eglMakeCurrent=bind)
        owner=NS(_context=ctypes.c_void_p(1));owner.make_current=lambda:bind('owner',None,None,owner._context)
        def free():
            events.append(('free',adapter._address(current[0])))
            if fail:raise RuntimeError('native failure')
        renderer=NS(_gl_context=owner,_mjr_context=NS(free=free))
        def close():
            self.assertIsNone(renderer._mjr_context)
            events.append(('close',adapter._address(current[0])))
        renderer.close=close
        context=NS(_native_rgb=renderer,_pixels=np.ones((2,2,3)),con=renderer._mjr_context,gl_ctx=owner)
        return context,egl,events,current

    def test_gpu_resources_freed_under_owner_before_gl_and_previous_restored(self):
        context,egl,events,current=self.release_fixture()
        with patch.dict(sys.modules,{'OpenGL':NS(EGL=egl)}):adapter._release(context)
        self.assertEqual(events,[('bind',1),('free',1),('close',1),('bind',2)])
        self.assertIsNone(context._native_rgb)
        self.assertIsNone(context.con)
        adapter.assert_healthy()

    def test_cleanup_failure_keeps_resources_alive_and_fails_closed(self):
        context,egl,events,current=self.release_fixture(True)
        with patch.dict(sys.modules,{'OpenGL':NS(EGL=egl)}):adapter._release(context)
        self.assertIs(adapter._quarantine[0],context)
        self.assertEqual(adapter._address(current[0]),2)
        self.assertNotIn(('close',1),events)
        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):adapter.assert_healthy()


if __name__=='__main__':unittest.main()
