"""RoboCasa365 native data with official fixed-interval sampling."""
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')

def history_indices(current, frames, interval=5):
    if current < 0 or frames < 1 or interval < 1:
        raise ValueError((current,frames,interval))
    # Official TemporalJitter(enable_jitter=False), including start padding.
    return [max(0,int(current)-i*int(interval)) for i in range(frames-1,-1,-1)]

def sample_current(length,rng):
    # Official map-style training can anchor at every episode frame, not only
    # multiples of the inference action-execution interval.
    return int(rng.integers(length))

class DemoStream:
    """Stateless index -> sample; restoring cursor exactly restores sampling."""
    def __init__(self, cfg):
        import pyarrow  # Load before the existing reader's native dependencies.
        from omegaconf import OmegaConf
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
        self.cfg = cfg
        if cfg['enable_jitter'] or cfg['training_anchor']!='all_episode_frames':
            raise ValueError('This protocol requires official fixed-interval, all-frame anchors')
        data_cfg = OmegaConf.create(cfg['reader'])
        data_cfg.data_mix = 'pi05_robocasa365_pretrain_all'
        # The base registry uses the original 100 demos per task, no A/B holdout.
        data_cfg.pop('history_split', None)
        self.mixture = get_vla_dataset(data_cfg, mode='train', seed=cfg['seed'])
        self.population = [{'name':d.dataset_name,'episodes':len(d.trajectory_ids),'frames':len(d)} for d in self.mixture.datasets]
        if len(self.population) != 300 or sum(x['episodes'] for x in self.population) != 30000:
            raise ValueError('Expected official 300 task x 100 demo population')

    def sample(self, index, *, current=None, frames=None):
        cfg = self.cfg
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], int(index)]))
        mixture = self.mixture
        task_id = int(rng.choice(len(mixture.datasets), p=mixture.dataset_sampling_weights))
        dataset = mixture.datasets[task_id]
        demo_id = int(rng.choice(len(dataset.trajectory_ids), p=mixture.trajectory_sampling_weights[task_id]))
        episode = int(dataset.trajectory_ids[demo_id])
        length = int(dataset.trajectory_lengths[demo_id])
        step = sample_current(length,rng) if current is None else int(current)
        if not 0 <= step < length:
            raise ValueError((step, length))
        times = history_indices(step, frames or cfg['history_frames'],cfg['history_interval'])
        units = [self.read(dataset, episode, t) for t in times]
        return {'units':units,'index':int(index),'dataset':dataset.dataset_name,
                'episode':episode,'length':length,'times':times,'task_index':task_id}

    @staticmethod
    def read(dataset, episode, step):
        raw = dataset.get_step_data(episode, step)
        sample = dataset._pack_sample(dataset.transforms(raw))
        return {'image':[np.asarray(x) for x in sample['image']],
                'state':np.asarray(sample['state'], dtype=np.float32),
                'action':np.asarray(sample['action'], dtype=np.float32),
                'lang':sample['lang']}

    def sequence(self, index=0, calls=7):
        selected = self.sample(index)
        dataset = self.mixture.datasets[selected['task_index']]
        start = 0
        if selected['length'] <= 5*(calls-1):
            raise ValueError('Selected episode too short for cache validation')
        units = [self.read(dataset, selected['episode'], start+5*i) for i in range(calls)]
        return units, {k:v for k,v in selected.items() if k!='units'}

class Processor:
    def __init__(self, cfg):
        from openpi import transforms
        from openpi.models import model, tokenizer
        from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
        from openpi.shared import normalize
        self.cfg = cfg
        self.stats = normalize.load(cfg['norm_assets'])
        self.tokenizer = tokenizer.PaligemmaTokenizer(cfg['max_token_len'])
        self.inputs = transforms.compose([
            RobocasaInputs(action_dim=32, model_type=model.ModelType.PI05),
            transforms.Normalize(self.stats, use_quantiles=False),
            transforms.ResizeImages(224,224),
            transforms.TokenizePrompt(self.tokenizer, discrete_state_input=cfg['discrete_state_input']),
            transforms.PadStatesAndActions(32),
        ])
        self.outputs = transforms.compose([transforms.Unnormalize(self.stats, use_quantiles=False), RobocasaOutputs()])

    def unit(self, unit, *, actions=False):
        if len(unit['image']) != 3 or np.shape(unit['state']) != (16,):
            raise ValueError('Expected 3 cameras and 16D native RoboCasa state')
        value = {'observation/image':np.asarray(unit['image'][0]),
                 'observation/wrist_image':np.asarray(unit['image'][1]),
                 'observation/right_image':np.asarray(unit['image'][2]),
                 'observation/state':np.asarray(unit['state']).copy(),'prompt':str(unit['lang'])}
        if actions:
            action = np.asarray(unit['action'], dtype=np.float32)
            if action.shape != (50,12):
                raise ValueError(('native action shape', action.shape))
            value['actions'] = action.copy()
        return self.inputs(value)

    def batch(self, samples):
        from openpi.models.model import Observation
        processed = [[self.unit(unit, actions=t==len(sample['units'])-1)
                      for t,unit in enumerate(sample['units'])] for sample in samples]
        images = {key:np.stack([np.stack([u['image'][key] for u in row]) for row in processed]) for key in CAMERAS}
        obs = Observation.from_dict({'image':images,
              'image_mask':{key:np.ones(len(processed), dtype=bool) for key in CAMERAS},
              'state':np.stack([row[-1]['state'] for row in processed]),
              'tokenized_prompt':np.stack([np.stack([u['tokenized_prompt'] for u in row]) for row in processed]).astype(np.int32),
              'tokenized_prompt_mask':np.stack([np.stack([u['tokenized_prompt_mask'] for u in row]) for row in processed])})
        return obs, np.stack([row[-1]['actions'] for row in processed])

    def online(self, units):
        # Reuse exactly the training transform, but actions are not required.
        from openpi.models.model import Observation
        row = [self.unit(u) for u in units]
        return Observation.from_dict({'image':{key:np.stack([u['image'][key] for u in row])[None] for key in CAMERAS},
            'image_mask':{key:np.ones(1,dtype=bool) for key in CAMERAS},
            'state':row[-1]['state'][None],
            'tokenized_prompt':np.stack([u['tokenized_prompt'] for u in row])[None].astype(np.int32),
            'tokenized_prompt_mask':np.stack([u['tokenized_prompt_mask'] for u in row])[None]})

    def action(self, normalized):
        action = np.asarray(self.outputs({'actions':np.asarray(normalized)})['actions'], dtype=np.float32)
        if action.shape != (50,12) or not np.isfinite(action).all():
            raise ValueError('Invalid environment-space action chunk')
        return action

def fingerprint(obs, actions=None):
    import jax
    digest = hashlib.sha256()
    for value in jax.tree.leaves(obs) + ([] if actions is None else [actions]):
        array = np.asarray(value)
        digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()

def save_sample(path, sample):
    arrays = {}
    for t, unit in enumerate(sample['units']):
        for cam, image in enumerate(unit['image']): arrays[f'image_{t}_{cam}'] = image
        arrays[f'state_{t}'] = unit['state']; arrays[f'action_{t}'] = unit['action']
        arrays[f'lang_{t}'] = np.asarray(unit['lang'])
    arrays['metadata'] = np.asarray(json.dumps({k:v for k,v in sample.items() if k!='units'}))
    np.savez_compressed(path, **arrays)

def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data['metadata'].item())
        meta['units'] = [{'image':[data[f'image_{t}_{c}'] for c in range(3)],
                         'state':data[f'state_{t}'],'action':data[f'action_{t}'],
                         'lang':data[f'lang_{t}'].item()} for t in range(len(meta['times']))]
    return meta
