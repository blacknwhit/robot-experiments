"""Versioned opt-in recipes; original paper30k validation remains distinct."""
import json
from common import ROOT

PROFILE_8GPU70K = 'pri4r_8gpu70k_v1'
PROFILES = (PROFILE_8GPU70K,)


def load_profile(name):
    if name not in PROFILES:
        raise ValueError('Unknown Pri4R training profile: '+str(name))
    value=json.loads((ROOT/'configs/profiles'/f'{name}.json').read_text())
    if value.get('schema')!='pri4r_training_profile_v1' or value.get('name')!=name or value.get('variant')!='pri4r_paper':
        raise ValueError('Invalid Pri4R profile identity')
    if (value['world_size'],value['per_device_batch_size'],value['gradient_accumulation_steps'],
        value['expected_global_batch_size'],value['max_train_steps'],value['stop_after_steps'])!=(8,16,1,128,70000,70000):
        raise ValueError('8GPU70k profile must preserve world8/B16/acc1/global128 and the full70k schedule')
    return value
