"""Portable wrapper around the audited immutable rigid-trajectory generator."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from common import ROOT,activate,environment,sha256
def main():
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);p.add_argument('--stage',choices=('prepare','generate','verify'),required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);a=p.parse_args()
    h=a.home.resolve();gt=h/'data/gt_armobj_full_v1';manifest=h/'data/gt_inputs/manifest.json';scope=h/'data/training_scope.json'
    activate('query3d_v2');env=environment('query3d_v2')
    config=json.loads((ROOT/'configs/gt_sampling.json').read_text());config['asset_root']=str(h)
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    gt.mkdir(parents=True,exist_ok=True);cfg=gt/'sampling_config.json';text=json.dumps(config,indent=2)
    if cfg.exists():
        if json.loads(cfg.read_text())!=config:raise ValueError('GT config changed; use a new output directory')
    elif a.stage=='prepare':
        with cfg.open('x') as f:f.write(text)
    else:raise FileNotFoundError('Run --stage prepare once before launching generation shards')
    if a.stage=='prepare':
        subprocess.run([sys.executable,'-m','starVLA.pri4r.source_prepare','--scope',str(scope),'--dataset-root',str(h/'data/robocasa365'),
            '--output-data',str(h/'data/gt_inputs'),'--manifest',str(manifest)],env=env,check=True)
    elif a.stage=='generate':
        subprocess.run([sys.executable,'-m','starVLA.pri4r.armobj_full_generate','--manifest',str(manifest),'--config',str(cfg),'--output',str(gt),
            '--workers',str(a.workers),'--shard-index',str(a.shard_index),'--num-shards',str(a.num_shards)],env=env,check=True)
    else:
        from starVLA.pri4r.reader import episode_key,validate_episode,PointTrackStore
        entries=json.loads(manifest.read_text())['episodes'];frames=0;seen=set()
        for e in entries:
            key=episode_key(e['source_prefix'],e['source_episode_index'])
            if key in seen:raise ValueError('Duplicate episode')
            seen.add(key)
            meta,arrays=validate_episode(gt/key,verify_hashes=True)
            if meta['config_sha256']!=digest or meta['frames']!=e['length'] or meta['points']!=1024:raise ValueError('GT identity/shape differs: '+key)
            if meta['episode_key']!=key or meta['repo']!='nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos' or meta['revision']!='522e4ffa3bb9d9729f79b241c65fc9e589f2659c':raise ValueError('GT source identity differs: '+key)
            alignment=meta['alignment_evidence']
            if alignment.get('passed') is not True:raise ValueError('Source/current alignment failed')
            for field in ('source_prefix','source_episode_index','current_dataset_name','current_episode_index','length'):
                if alignment[field]!=e[field]:raise ValueError('GT episode alignment differs: '+field)
            PointTrackStore._close_arrays(arrays);frames+=meta['frames']
        if (len(seen),frames)!=(30000,27815381):raise ValueError('Incomplete GT scope')
        proof={'episodes':len(seen),'frames':frames,'config_sha256':digest,'manifest_sha256':sha256(manifest),'verified_all_array_hashes':True}
        (gt/'PORTABLE_GT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':main()
