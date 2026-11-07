"""Per-episode policy with the official bounded, periodic-flush lifecycle."""
import json
from pathlib import Path
import jax
import numpy as np
from openpi.models import model as model_api
from openpi.shared import nnx_utils
from data import Processor, fingerprint
from engine import make_config
from openpi import transforms
from openpi.policies.robocasa_policy import RobocasaOutputs

def empty_memory():
    return dict(memory_tokens=None,memory_kv_cache=None,memory_prefix_mask=None)

class StreamPolicy:
    def __init__(self,cfg,checkpoint,*,raw_params=None):
        self.cfg=dict(cfg)
        checkpoint=Path(checkpoint)
        if (checkpoint/'assets').is_dir(): self.cfg['norm_assets']=str(checkpoint/'assets')
        self.processor=Processor(self.cfg)
        # Policy returns only actions. The official Unnormalize is strict, so
        # select only action stats instead of requiring an unused output state.
        self.processor.outputs=transforms.compose([
            transforms.Unnormalize({'actions':self.processor.stats['actions']},use_quantiles=False),
            RobocasaOutputs(),
        ])
        params = raw_params if raw_params is not None else model_api.restore_params(checkpoint/'params')
        self.model=make_config(cfg).model.load(params,remove_extra_params=False)
        self.model.eval()
        self.sample=nnx_utils.module_jit(self.model.sample_actions)
        self.sessions={}

    def reset(self,session):
        self.sessions.pop(str(session),None)

    def predict(self,unit,*,session,call_index,noise_seed,noise=None):
        session=str(session)
        state=self.sessions.setdefault(session,{'next_call':0,'memory':empty_memory()})
        if call_index != state['next_call']:
            raise ValueError('Nonsequential call index; reset explicitly at each episode')
        flush = call_index % self.cfg['history_frames'] == 0
        memory=empty_memory() if flush else state['memory']
        obs=self.processor.online([unit])
        rng=jax.random.key(noise_seed)
        if noise is None: noise=jax.random.normal(rng,(1,self.cfg['action_horizon'],32))
        actions,memory=self.sample(rng,obs,noise=noise,memory=memory)
        actions=np.asarray(actions)
        state.update(next_call=call_index+1,memory=memory)
        metadata={'call_index':call_index,'flush':flush,'context_units':call_index%self.cfg['history_frames']+1,
                  'prefix_tokens':int(memory['memory_prefix_mask'].shape[-1]),'input_sha256':fingerprint(obs),
                  'noise_seed':noise_seed,'normalized_action_sha256':__import__('hashlib').sha256(actions.tobytes()).hexdigest()}
        return self.processor.action(actions[0]),metadata,actions
