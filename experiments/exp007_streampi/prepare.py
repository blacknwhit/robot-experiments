"""Resolve an existing resource set; never installs, downloads, or launches jobs."""
import argparse
import json
from pathlib import Path
from contract import ROOT,REPO,sha,validate,source_manifest,load,verify_assets,recipe_identity

def configure(args):
    cfg=json.loads((ROOT/'recipe_8gpu64_70k.json').read_text())
    scope=Path(args.workspace_root).resolve();run=(scope/args.run_id).resolve();run.relative_to(scope)
    if run==scope or len(Path(args.run_id).parts)!=1:raise ValueError('run-id must be a unique single directory name')
    if not args.run_id.startswith('zf-'):raise ValueError('Use zf- task/run names')
    resources=json.loads(Path(args.resources).read_text())
    for name,value in resources.items():
        p=Path(value).resolve();p.relative_to(scope)
        if name!='arrow_cache' and not p.exists():raise FileNotFoundError(p)
        resources[name]=str(p)
    reader=json.loads((ROOT/'reader_template.json').read_text())
    reader.update(data_root_dir=resources['data_root'],official_registry_file=resources['registry_file'],v3_arrow_cache_dir=resources['arrow_cache'])
    cfg.update(workspace_root=str(scope),run_root=str(run),mode=args.mode,reader=reader,
               base_checkpoint=resources['base_checkpoint'],norm_assets=resources['norm_assets'],tokenizer=resources['tokenizer'],
               batch_size=64,source_files=source_manifest())
    files=[Path(resources['tokenizer']),Path(resources['registry_file'])]
    files+=list(Path(resources['base_checkpoint']).rglob('*'))+list(Path(resources['norm_assets']).rglob('*.json'))
    for bundle in ('robocasa365-pretrain-atomic','robocasa365-pretrain-composite'):
        meta=Path(resources['data_root'])/bundle/'meta'
        files+=list(meta.glob('*.json'))+list(meta.glob('*.parquet'))+list((meta/'episodes').glob('*/*.parquet'))
    cfg['resource_manifest']={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files)) if p.is_file()}
    if not any(str(Path(resources['base_checkpoint'])) in p for p in cfg['resource_manifest']):raise ValueError('Empty base checkpoint')
    # The exact original JAX base used in the accepted engineering run.
    manifest={str(Path(p).relative_to(resources['base_checkpoint'])):record for p,record in cfg['resource_manifest'].items()
              if Path(resources['base_checkpoint']) in Path(p).parents}
    import hashlib
    cfg['base_tree_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if cfg['base_tree_sha256']!='737632cdece0b202ea7d35e32ad55a7c3f03f6c6606be41ed6c34d7996ca343f':
        raise ValueError('Base checkpoint differs from accepted original pi05_base; verify provenance before changing contract')
    validate(cfg)
    if args.mode=='formal':
        if not args.gpu_acceptance:raise ValueError('First complete the two actual eight-GPU smoke runs and compare_runs.py')
        gate_path=Path(args.gpu_acceptance).resolve();gate_path.relative_to(scope)
        gate=json.loads(gate_path.read_text())
        if not gate.get('passed') or gate.get('physical_gpu_count')!=8 or gate.get('recipe_identity')!=recipe_identity(cfg):
            raise ValueError('GPU acceptance does not cover this exact formal recipe')
        cfg['gpu_acceptance']=gate
    out=Path(args.out).resolve();out.relative_to(scope);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(cfg,f,indent=2)
    print(json.dumps({'config':str(out),'mode':args.mode,'gpus':8,'global_batch':64,'updates':70000,'resources':len(cfg['resource_manifest'])}))

def check_data(args):
    cfg=load(args.config);verify_assets(cfg)
    from environment import setup
    setup(cfg,cpu=True)
    from data import DemoStream,Processor,fingerprint
    stream=DemoStream(cfg);processor=Processor(cfg)
    rows=[]
    for i in range(4):
        s=stream.sample(i);batch=processor.batch([s]);rows.append({k:v for k,v in s.items() if k!='units'}|{'input_sha256':fingerprint(*batch)})
    import numpy as np
    obs,actions=processor.batch([stream.sample(i) for i in range(64)])
    assert actions.shape==(64,50,32) and obs.tokenized_prompt.shape==(64,3,200)
    assert all(v.shape==(64,3,224,224,3) for v in obs.images.values())
    assert np.isfinite(actions).all()
    from contract import identity
    result={'passed':True,'identity':identity(cfg),'tasks':len(stream.population),'demos':sum(x['episodes'] for x in stream.population),'samples':rows,
            'real_global_batch':64,'batch_input_sha256':fingerprint(obs,actions),'image_shapes':{k:list(v.shape) for k,v in obs.images.items()},
            'prompt_shape':list(obs.tokenized_prompt.shape),'action_shape':list(actions.shape)}
    p=Path(cfg['run_root'])/'DATA_ACCEPTED.json'
    with p.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('configure');c.add_argument('--resources',required=True);c.add_argument('--workspace-root',required=True)
    c.add_argument('--run-id',required=True);c.add_argument('--mode',choices=['smoke','formal'],required=True);c.add_argument('--out',required=True)
    c.add_argument('--gpu-acceptance')
    d=sub.add_parser('check-data');d.add_argument('--config',required=True)
    a=p.parse_args();configure(a) if a.command=='configure' else check_data(a)
if __name__=='__main__':main()
