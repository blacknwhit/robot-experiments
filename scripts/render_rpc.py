"""Private-pipe RGB rendering: CPU-only simulators, one EGL-owning process."""
import gc
import pickle
import time
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from types import SimpleNamespace
import weakref

import numpy as np
from render_scene_packet import ScenePacket

TRANSPORT = 'mujoco331_complete_mjvscene_pipe_v1'
_connection = None
_contexts = weakref.WeakSet()
_registered_contexts = {}
_next_id = 0
_defer_enabled = False
_in_construction = False
_construction_checked = False
_deferred_frames = 0


@contextmanager
def construction_scope():
    """Preserve both resets and all CPU/RNG work; omit only discarded GPU draws."""
    global _in_construction, _construction_checked
    if not _defer_enabled:
        yield
        return
    if _in_construction:
        raise RuntimeError('Nested simulator construction is unsupported')
    if not _construction_checked:
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        from robocasa.environments.kitchen.kitchen import Kitchen
        pinned = [(RoboCasaGymEnv.__init__, 'd091b31acd56c1b1a5fe6d9843f22a8037032ea7b1828632e9751a91ddd3e48c'),
                  (RoboCasaGymEnv._create_obs_and_action_space, 'e1dba88de533ec61e0f3bd59e2884721c10ff88fcf57b842cc3895dd119d2d7e'),
                  (Kitchen.__init__, '18317777a5f9fb34d027d7208e3ca403704888c6a1cc4b96b5814ffca80dc0a9')]
        for method, expected in pinned:
            source = textwrap.dedent(inspect.getsource(method))
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                raise RuntimeError('Unvalidated constructor; use --initial-rgb all')
        _construction_checked = True
    _in_construction = True
    try:
        yield
    finally:
        _in_construction = False


def validate_final_reset(env):
    if _defer_enabled and env.unwrapped.env.hard_reset is not True:
        raise RuntimeError('Deferred construction requires a native hard reset')


def call(*message):
    _connection.send(message)
    response = _connection.recv()
    if not response['ok']:
        raise RuntimeError('Render owner failed: ' + response['error'])
    return response.get('value')


def _init(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
    import mujoco
    global _next_id
    if not offscreen:
        raise ValueError('Only offscreen RGB is supported')
    if sim.model._model.nplugin:
        raise ValueError('Native visualization plugins are not supported by scene IPC')
    self.sim, self.model, self.data = sim, sim.model, sim.data
    self.offscreen, self.device_id = True, device_id
    self._remote_id = _next_id
    _next_id += 1
    self._registered = False
    self._pixels = None
    self._proof = None
    self.con = self.gl_ctx = None
    sim.forward()
    sim.add_render_context(self)
    self.scn = mujoco.MjvScene(sim.model._model, maxgeom=10000)
    self._packet = ScenePacket(self.scn)
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
    global _deferred_frames
    if segmentation or self.pert.active or self.pert.active2:
        raise ValueError('Only unperturbed RGB is supported')
    if not self._registered and not _in_construction:
        # MuJoCo's native serializer preserves the compiled model. No XML recompile.
        call('register', self._remote_id, pickle.dumps(self.model._model, protocol=5))
        self._registered = True
        _registered_contexts[self._remote_id] = self
    if camera_id is not None:
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE if camera_id == -1 else mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = camera_id
    # Exact update_scene operation used by mujoco.Renderer, without any GL.
    mujoco.mjv_updateScene(self.model._model, self.data._data, self.vopt, None,
                          self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
    if _in_construction:
        # Constructor observations are discarded; the source-pinned final hard
        # reset below constructs the scored observation with real RGB rendering.
        self._pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self._size = (width, height)
        _deferred_frames += 1
        return
    response = call('render', self._remote_id, width, height, self._packet.capture())
    self._pixels = response['pixels']
    self._proof = response['proof']
    self._size = (width, height)
    self.con = SimpleNamespace(offSamples=self._proof['msaa'])


def _read(self, width, height, depth=False, segmentation=False):
    if depth or segmentation or self._pixels is None or self._size != (width, height):
        raise ValueError('Unmatched or unsupported RGB read')
    return self._pixels.copy()


def release(context):
    if getattr(context, '_registered', False):
        call('release', context._remote_id)
        context._registered = False
        _registered_contexts.pop(context._remote_id, None)
    context._pixels = None


def retain_environments(environments):
    live = [env.unwrapped.env.sim._render_context_offscreen for env in environments]
    for context in list(_registered_contexts.values()):
        if context not in live:
            release(context)
    if any(not getattr(context, '_proof', None) for context in live):
        raise RuntimeError('Missing render-owner proof')
    return [context._proof for context in live]


def release_all():
    for context in list(_registered_contexts.values()):
        release(context)


def install(connection, defer_construction=False):
    global _connection, _defer_enabled
    _connection = connection
    _defer_enabled = defer_construction
    import simulator_renderer
    import mujoco
    from robosuite.utils.binding_utils import MjRenderContext
    if mujoco.__version__ != '3.3.1':
        raise RuntimeError('Pinned MuJoCo required')
    MjRenderContext.__init__ = _init
    MjRenderContext.render = _render
    MjRenderContext.read_pixels = _read
    MjRenderContext.update_offscreen_size = lambda self, width, height: release(self)
    # Explicit safe cleanup only. Do not issue blocking RPC from a finalizer.
    MjRenderContext.__del__ = lambda self: None
    def unsupported_upload(*args, **kwargs):
        raise RuntimeError('Runtime GPU asset uploads require an explicit RPC implementation')
    for name in ('upload_texture', 'upload_mesh', 'upload_hfield'):
        if hasattr(MjRenderContext, name):
            setattr(MjRenderContext, name, unsupported_upload)
    MjRenderContext._native_rgb_version = simulator_renderer.VERSION
    simulator_renderer.install = lambda: simulator_renderer.VERSION
    simulator_renderer.retain_environments = retain_environments
    simulator_renderer.release_all = release_all


class RenderOwner:
    """Called only by the controller main thread; connections are internal pipes."""
    def __init__(self):
        self.contexts = {}
        self.render_seconds = 0.
        self.frames = 0
        self.registered_models = 0
        self.peak_contexts = 0
        self.timing = dict(register_seconds=0., context_seconds=0., scene_copy_seconds=0., draw_seconds=0.)

    def request(self, worker, message):
        import mujoco
        import simulator_renderer
        op, local_id, *args = message
        key = (worker, local_id)
        if op == 'register':
            if key in self.contexts:
                raise ValueError('Duplicate render context')
            start = time.perf_counter()
            model = pickle.loads(args[0])
            self.timing['register_seconds'] += time.perf_counter()-start
            if not isinstance(model, mujoco.MjModel):
                raise ValueError('Invalid compiled model')
            self.contexts[key] = SimpleNamespace(model=model, _native_rgb=None, _pixels=None,
                                                 con=None, gl_ctx=None, size=None)
            self.registered_models += 1
            self.peak_contexts = max(self.peak_contexts, len(self.contexts))
            return None
        context = self.contexts[key]
        if op == 'release':
            simulator_renderer._release(context)
            simulator_renderer.assert_healthy()
            del self.contexts[key]
            return None
        if op != 'render':
            raise ValueError('Unknown render request')
        width, height, packet = args
        if context._native_rgb is None or context.size != (width, height):
            start = time.perf_counter()
            simulator_renderer._release(context)
            simulator_renderer.assert_healthy()
            context._native_rgb = mujoco.Renderer(context.model, height=height, width=width, max_geom=10000)
            context.size = (width, height)
            context.packet = ScenePacket(context._native_rgb.scene)
            self.timing['context_seconds'] += time.perf_counter()-start
        renderer = context._native_rgb
        if type(renderer._gl_context).__module__ != 'mujoco.egl':
            raise RuntimeError('Render owner must use EGL')
        start = time.perf_counter()
        context.packet.restore(packet)
        self.timing['scene_copy_seconds'] += time.perf_counter()-start
        start = time.perf_counter()
        pixels = renderer.render(out=np.zeros((height, width, 3), dtype=np.uint8))[::-1].copy()
        self.timing['draw_seconds'] += time.perf_counter()-start
        self.frames += 1
        return {'pixels': pixels, 'proof': {'adapter': simulator_renderer.VERSION, 'engine': 'mujoco',
                'backend': 'egl', 'msaa': int(renderer._mjr_context.offSamples), 'max_geom': 10000,
                'gl_context_type': 'mujoco.egl.GLContext', 'transport': TRANSPORT}}

    def close(self):
        import simulator_renderer
        for context in list(self.contexts.values()):
            simulator_renderer._release(context)
        self.contexts.clear()
        gc.collect()
        simulator_renderer.assert_healthy()
