"""Model-neutral requests, strict output contracts, and isolated episode state.

Adapters live with model code. They never own scenes, scoring, or scheduling.
"""
from contextlib import nullcontext
from dataclasses import dataclass, field
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from . import API_VERSION

ACTION_LAYOUT = 'robocasa_openpi_environment12_v1'
RNG_MODE = 'stateless_request_v1'
SEED_VERSION = 'sha256_u32_v1'


@dataclass(frozen=True)
class LoadContext:
    config: Path
    checkpoint: Path
    device: str
    options: dict
    backend: str
    backend_version: str
    source_files: frozenset
    assets: dict


@dataclass(frozen=True)
class EpisodeContext:
    run_id: str
    task: str
    episode: int


@dataclass(frozen=True)
class PolicyRequest:
    images: tuple
    state: np.ndarray
    language: str
    seed: int
    episode: EpisodeContext
    replan: int
    robot_tag: str

    def starvla_example(self):
        """Retain the native PI0.5 example, without transport/session metadata."""
        return {'image': list(self.images), 'state': self.state, 'lang': self.language,
                'robot_tag': self.robot_tag, '__pi05_rng_seed': self.seed}


class ModelAdapter:
    """Implement predict; complete loading and normalization remain model-owned.

    Stateful adapters must return a distinct state from begin_episode and use
    only that state for episode memory. Never reset another environment's state.
    Explicit-RNG concurrent adapters must own their RNG and shared GPU buffers.
    """
    state_mode = 'stateless'
    rng_mode = 'managed_global'
    max_concurrency = 1
    backend = 'native'
    backend_version = 'model_native_v1'
    precision = 'model_native'

    def begin_episode(self, context):
        if self.state_mode != 'stateless':
            raise NotImplementedError('Stateful adapter must allocate episode-local state')
        return None

    def predict(self, request, state):
        raise NotImplementedError

    def end_episode(self, context, state, reason):
        pass

    def close(self):
        pass

    @property
    def extra_metadata(self):
        return {}


def request_seed(task, episode, replan):
    fields = (SEED_VERSION, '7', task, '0', str(episode), str(replan))
    return int.from_bytes(hashlib.sha256('\0'.join(fields).encode()).digest()[:4], 'big')


def _readonly(array):
    view = array.view()
    view.setflags(write=False)
    return view


def decode_request(payload, context, expected_replan):
    wire = payload.get('context', {})
    if wire != {**vars(context), 'replan': expected_replan}:
        raise ValueError('Foreign, duplicated, or out-of-order episode request')
    examples = payload.get('examples')
    if not isinstance(examples, list) or len(examples) != 1:
        raise ValueError('Adapter protocol requires exactly one observation')
    sample = examples[0]
    if not isinstance(sample, dict):raise ValueError('Observation must be a mapping')
    seed = sample.get('__pi05_rng_seed')
    if type(seed) is not int or seed != request_seed(context.task, context.episode, expected_replan):
        raise ValueError('Request RNG seed differs from fixed benchmark protocol')
    images = sample.get('image')
    if not isinstance(images, (list, tuple)) or len(images) != 3:
        raise ValueError('Expected left, wrist, right RGB images')
    checked = []
    for image in images:
        array = np.asarray(image)
        if array.shape != (256, 256, 3) or array.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB images with shape256x256x3')
        checked.append(_readonly(array))
    state = np.asarray(sample.get('state'))
    if state.shape != (16,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError('Expected finite float32 state16')
    if not isinstance(sample.get('lang'), str) or not isinstance(sample.get('robot_tag'), str):
        raise ValueError('Language and robot tag must be strings')
    return PolicyRequest(tuple(checked), _readonly(state), sample['lang'], seed, context,
                         expected_replan, sample['robot_tag'])


@dataclass
class _Session:
    lock: object = field(default_factory=threading.RLock)
    context: object = None
    state: object = None
    requests: int = 0
    failed: bool = False


class ManagedPolicy:
    """Enforce connection-owned lifecycle without introducing model assumptions."""
    def __init__(self, adapter, identity, task_horizons, *, seed_scope=None):
        self.adapter = adapter
        self.task_horizons = {'robocasa/' + k.removeprefix('robocasa/'): v for k,v in task_horizons.items()}
        self.seed_scope = seed_scope
        self.sessions = {}
        self.closed = False
        self.failure = None
        self.active_contexts = {}
        self.active_states = set()
        self.lock = threading.RLock()
        self.model_lock = threading.RLock()
        self.model_slots = threading.BoundedSemaphore(adapter.max_concurrency)
        if adapter.state_mode not in ('stateless','per_episode'):
            raise ValueError('Unsupported state capability')
        if adapter.rng_mode not in ('explicit_per_request','managed_global'):
            raise ValueError('Unsupported RNG capability')
        if type(adapter.max_concurrency) is not int or not 1 <= adapter.max_concurrency <= 16:
            raise ValueError('Adapter concurrency must be1..16')
        if adapter.rng_mode == 'managed_global' and (adapter.max_concurrency != 1 or seed_scope is None):
            raise ValueError('Managed global RNG requires serialized execution and a seed scope')
        for key in ('backend','backend_version','state_mode','rng_mode','max_concurrency'):
            expected_key = {'backend':'inference_backend','backend_version':'inference_backend_version'}.get(key,key)
            if identity.get(expected_key) != getattr(adapter,key):
                raise ValueError('Adapter capability disagrees with model spec: ' + key)
        proof = copy.deepcopy(adapter.load_proof)
        if (proof.get('strict') is not True or proof.get('all_keys_present') is not True
                or proof.get('all_loaded_values_exact_at_model_dtype') is not True
                or proof.get('checkpoint_keys_dropped') != 0):
            raise ValueError('Adapter must provide complete model loading evidence')
        metadata = {**identity, 'model_adapter_api_version': API_VERSION,
            'action_output_space':'environment','action_layout':ACTION_LAYOUT,'action_chunk_size':50,
            'action_dim':12,'state_dim':16,'image_observation_keys':['observation/image','observation/wrist_image','observation/right_image'],
            'policy_rng_mode':RNG_MODE,'policy_seed_version':SEED_VERSION,'strict_full_state':proof,
            'training_precision_mode':str(adapter.precision)}
        extra = copy.deepcopy(adapter.extra_metadata)
        if not isinstance(extra,dict) or set(extra) & set(metadata):
            raise ValueError('Extra model metadata must not override protocol/identity fields')
        metadata.update(extra)
        # Freeze a JSON-only handshake snapshot; episode counters never enter it.
        self.metadata = json.loads(json.dumps(metadata,allow_nan=False))

    def connect(self):
        key = uuid.uuid4().hex
        with self.lock:
            if self.closed or self.failure:raise ValueError('Model service is closed or unhealthy')
            self.sessions[key] = _Session()
        return key

    def _model_scope(self):
        return self.model_lock if self.adapter.max_concurrency == 1 else self.model_slots

    def _episode(self, payload):
        if not isinstance(payload,dict) or set(payload) != {'run_id','task','episode'}:
            raise ValueError('Episode context must contain run_id, task, episode')
        if not isinstance(payload['run_id'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',payload['run_id']):
            raise ValueError('Invalid run namespace')
        if payload['task'] not in self.task_horizons or type(payload['episode']) is not int or not 0 <= payload['episode'] < 50:
            raise ValueError('Unknown task or global episode ID')
        return EpisodeContext(**payload)

    def _end(self, sid, session, reason):
        context, state = session.context, session.state
        if context is None:return None
        try:
            with self._model_scope():self.adapter.end_episode(context,state,reason)
        except BaseException as error:
            self.failure = repr(error);session.failed = True
            raise
        with self.lock:
            self.active_contexts.pop((context.run_id,context.task,context.episode),None)
            if state is not None:self.active_states.discard(id(state))
        session.context = session.state = None
        return {'api_version':API_VERSION,'session_id':sid,'context':vars(context),
                'requests':session.requests,'reason':reason,'state_released':True}

    def dispatch(self, sid, message):
        with self.lock:session = self.sessions.get(sid)
        if session is None:raise ValueError('Closed adapter connection')
        if not isinstance(message,dict):raise ValueError('Wire message must be a mapping')
        kind, payload = message.get('type'), message.get('payload',{})
        with session.lock:
            with self.lock:
                if self.closed or self.failure or self.sessions.get(sid) is not session:raise ValueError('Closed or unhealthy adapter connection')
            if kind == 'adapter_begin':
                if session.context is not None:raise ValueError('End previous episode before beginning another')
                context = self._episode(payload)
                key = (context.run_id,context.task,context.episode)
                with self.lock:
                    if key in self.active_contexts:raise ValueError('Duplicate active task/episode context')
                    self.active_contexts[key] = sid
                state = None
                try:
                    with self._model_scope():
                        scope = self.seed_scope(request_seed(context.task,context.episode,0)) if self.adapter.rng_mode == 'managed_global' else nullcontext()
                        with scope:state = self.adapter.begin_episode(context)
                    if self.adapter.state_mode == 'stateless' and state is not None:
                        raise ValueError('Stateless adapter must not allocate hidden episode state')
                    if self.adapter.state_mode == 'per_episode' and state is None:
                        raise ValueError('Stateful adapter did not allocate episode state')
                    with self.lock:
                        if state is not None and id(state) in self.active_states:
                            raise ValueError('Adapter reused another episode state object')
                        if state is not None:self.active_states.add(id(state))
                except BaseException:
                    with self.lock:
                        shared = state is not None and id(state) in self.active_states
                        self.active_contexts.pop(key,None)
                    if state is not None and not shared:
                        with self._model_scope():self.adapter.end_episode(context,state,'begin_failed')
                    raise
                session.context,session.state = context,state
                session.requests,session.failed = 0,False
                return {'api_version':API_VERSION,'session_id':sid,'context':vars(context),'begun':True}
            if kind in ('adapter_end','adapter_abort'):
                if session.context is None:raise ValueError('No active episode')
                if payload.get('context') != vars(session.context):raise ValueError('End context mismatch')
                if kind == 'adapter_end':
                    if session.failed or payload.get('requests') != session.requests:
                        raise ValueError('Cannot commit a failed or mismatched episode session')
                    completed = payload.get('completed')
                    if type(completed) is not bool:raise ValueError('Episode completion must be explicit')
                    reason = 'completed' if completed else 'diagnostic_limit'
                else:reason = 'aborted'
                return self._end(sid,session,reason)
            if kind != 'adapter_infer':raise ValueError('Unsupported adapter operation')
            if session.context is None or session.failed:raise ValueError('Begin a healthy episode before inference')
            if session.requests >= (self.task_horizons[session.context.task]+4)//5:
                raise ValueError('Inference exceeds native episode horizon')
            try:
                request = decode_request(payload,session.context,session.requests)
                with self._model_scope():
                    scope = self.seed_scope(request.seed) if self.adapter.rng_mode == 'managed_global' else nullcontext()
                    with scope:actions = self.adapter.predict(request,session.state)
                    if not isinstance(actions,np.ndarray) or actions.shape != (50,12) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                        raise ValueError('Adapter must return finite float32 environment actions[50,12]')
                    # A model may return a reusable output buffer. Own the wire copy.
                    actions = np.array(actions,copy=True,order='C')[None]
                session.requests += 1
                return {'actions':actions,'adapter_request':{'session_id':sid,'replan':request.replan,'accepted':True}}
            except BaseException:
                session.failed = True
                raise

    def disconnect(self, sid):
        with self.lock:session = self.sessions.get(sid)
        if session is None:return
        with session.lock:
            self._end(sid,session,'connection_closed')
            with self.lock:self.sessions.pop(sid,None)

    def close(self):
        with self.lock:self.closed=True
        errors=[]
        for sid in list(self.sessions):
            try:self.disconnect(sid)
            except BaseException as error:errors.append(repr(error))
        try:
            with self._model_scope():self.adapter.close()
        except BaseException as error:errors.append(repr(error))
        if errors:raise RuntimeError('Model adapter cleanup failed: ' + '; '.join(errors))
