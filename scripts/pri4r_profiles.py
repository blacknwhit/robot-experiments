"""Versioned opt-in recipes; original paper30k validation remains distinct."""
import json
from common import ROOT

PROFILE_8GPU70K = 'pri4r_8gpu70k_v1'
PROFILE_8GPU70K_B8 = 'pri4r_8gpu70k_b8_v2'
PROFILES = (PROFILE_8GPU70K, PROFILE_8GPU70K_B8)


def load_profile(name):
    if name not in PROFILES:
        raise ValueError('Unknown Pri4R training profile: '+str(name))
    value=json.loads((ROOT/'configs/profiles'/f'{name}.json').read_text())
    if value.get('schema')!='pri4r_training_profile_v1' or value.get('name')!=name or value.get('variant')!='pri4r_paper':
        raise ValueError('Invalid Pri4R profile identity')
    micro,batch=(8,64) if name==PROFILE_8GPU70K_B8 else (16,128)
    if (value['world_size'],value['per_device_batch_size'],value['gradient_accumulation_steps'],
        value['expected_global_batch_size'],value['max_train_steps'],value['stop_after_steps'])!=(8,micro,1,batch,70000,70000):
        raise ValueError('8GPU70k profile must preserve its versioned batch and full70k schedule')
    return value


def expected_global_batch(config):
    """Use the registered recipe, never an unchecked user-edited batch field."""
    portable=config['portable']
    if portable['variant']!='pri4r_paper':return 64
    name=portable.get('training_profile')
    return load_profile(name)['expected_global_batch_size'] if name else 128
