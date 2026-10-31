"""Resolve one immutable recipe for a new cluster; never launches training."""
import argparse, hashlib, json, re
from pathlib import Path
import yaml
from common import ROOT, VARIANTS, POSITION_VARIANTS, runtime, sha256

def render(variant, home, run_id, world, stop=70000, smoke=False, engineering=False, formal=False,
           experiment_root=None):
    if world not in (2,4,8) or not 1 <= stop <= 70000:
        raise ValueError('Supported world sizes: 2/4/8; stop must be 1..70000')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id): raise ValueError('run-id must be a simple directory name')
    if sum((smoke, engineering, formal)) > 1:
        raise ValueError('Choose exactly one training mode')
    if variant in POSITION_VARIANTS:
        if not (smoke or engineering or formal) or stop > (10 if smoke else (1500 if engineering else 70000)):
            raise ValueError('V3 requires explicit --smoke <=10, --engineering <=1500, or --formal <=70000')
    if variant == 'pri4r_paper' and (not (smoke or formal) or stop > (10 if smoke else 30000)):
        raise ValueError('Pri4R requires --smoke --stop <=10 or --formal --stop <=30000')
    home=Path(home).resolve(); r=runtime(variant).resolve()
    output_home = home
    if experiment_root is not None:
        if variant not in POSITION_VARIANTS and variant != 'pri4r_paper':
            raise ValueError('--experiment-root is scoped to the new A/B/C experiment')
        output_home = Path(experiment_root).resolve()
        if output_home == home:
            raise ValueError('Use a distinct experiment output root, not the shared resource home')
    gt=home/('data/gt_pri4r_paper' if variant == 'pri4r_paper' else 'data/gt_armobj_full_v1')
    gt_hash=None
    if (gt/'sampling_config.json').exists():
        config=json.loads((gt/'sampling_config.json').read_text())
        gt_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    replacements={'OPENPI_ROOT':r/'playground/Code/openpi_robocasa365','BASE_WEIGHTS':home/'models/pi05_base_pytorch/model.safetensors',
        'NORM_ASSETS':home/'models/robocasa_norm','DATA_ROOT':home/'data/robocasa365',
        'REGISTRY_FILE':ROOT/'assets/dataset_registry.py','ARROW_CACHE':home/'cache/arrow',
        'GT_ROOT':gt,'GT_CONFIG_SHA256':gt_hash or 'GENERATE_GT_FIRST', 'RUN_ID':run_id,
        'RUN_ROOT':output_home/('smoke_runs' if smoke else ('engineering_runs' if engineering else 'runs'))}
    text=(ROOT/'configs'/f'{variant}.yaml').read_text()
    for key,val in replacements.items():text=text.replace('${'+key+'}',str(val))
    if '${' in text:raise ValueError('Unresolved template variable')
    c=yaml.safe_load(text)
    c['datasets']['vla_data']['per_device_batch_size']=(128 if variant == 'pri4r_paper' else 64)//world
    c['trainer']['stop_after_steps']=stop
    if smoke:c['trainer'].update(save_interval=1,logging_frequency=1)
    c['portable']={'variant':variant,'world_size':world,'smoke':smoke}
    if variant == 'pri4r_paper':
        c['portable']['resource_home'] = str(home)
        if smoke:
            c['datasets']['vla_data']['pri4r_smoke_manifest'] = str(gt/'input_manifest.json')
            c['datasets']['vla_data']['num_workers'] = 0
            c['datasets']['vla_data']['prefetch_factor'] = None
    if experiment_root is not None:
        c['portable'].update(resource_home=str(home), experiment_root=str(output_home))
    if engineering:
        c['portable']['engineering'] = True
    if formal:
        c['portable']['formal'] = True
        if variant in POSITION_VARIANTS:
            c['experiment'].update(phase='formal_candidate', formal_training_authorized=True)
    # Same immutable code identity used by v2 checkpoint contracts on fresh/resumed runs.
    c['trajectory_release_sha256']=sha256(ROOT/'provenance/source_files.sha256')
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--home',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--gpus',type=int,choices=(2,4,8),required=True)
    p.add_argument('--stop',type=int,default=70000);p.add_argument('--smoke',action='store_true')
    p.add_argument('--engineering',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--formal',action='store_true',help='Explicitly prepare a fresh formal candidate recipe; never starts a job')
    p.add_argument('--experiment-root', type=Path,
                   help='A/B/C outputs only; --home still supplies existing data, base, GT and caches')
    a=p.parse_args();c=render(a.variant,a.home,a.run_id,a.gpus,a.stop,a.smoke,a.engineering,a.formal,
                            experiment_root=a.experiment_root)
    if a.out.exists():raise FileExistsError('Refusing to overwrite an existing resolved recipe')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(yaml.safe_dump(c,sort_keys=False))
    print(a.out.resolve())
if __name__=='__main__':main()
