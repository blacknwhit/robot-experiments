"""Paper-style surface tracks; bounded four-demo preparation is the default.

Full generation is only selected explicitly with --full on the training cluster.
This does not launch GPU training and never changes an existing sampling cache.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT,activate,environment,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',type=Path,required=True)
    p.add_argument('--stage',choices=('prepare','generate','verify'),required=True)
    p.add_argument('--full',action='store_true')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--input-manifest',type=Path,help='Use an already source-aligned manifest instead of scanning data')
    p.add_argument('--asset-root',type=Path,help='Existing asset tree root; defaults to --home')
    a=p.parse_args();h=a.home.resolve();gt=h/'data/gt_pri4r_paper';manifest=gt/'input_manifest.json'
    config=json.loads((ROOT/'configs/pri4r_paper_gt.json').read_text());config['asset_root']=str((a.asset_root or h).resolve())
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    cfg=gt/'sampling_config.json';gt.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':
        if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Use a new home for changed GT sampling')
        if not cfg.exists():cfg.write_text(json.dumps(config,indent=2)+'\n')
        if a.input_manifest:
            value=json.loads(a.input_manifest.read_text())
            if manifest.exists() and json.loads(manifest.read_text())!=value:raise ValueError('Existing GT manifest differs')
            if not manifest.exists():manifest.write_text(json.dumps(value,indent=2)+'\n')
        else:
            cmd=[sys.executable,'-m','starVLA.pri4r.source_prepare','--scope',str(h/'data/training_scope.json'),
                 '--dataset-root',str(h/'data/robocasa365'),'--output-data',str(h/'data/pri4r_source_inputs'),
                 '--manifest',str(manifest)]
            if not a.full:cmd.append('--sample')
            subprocess.run(cmd,env=environment('pri4r_paper'),check=True)
    if not cfg.is_file() or json.loads(cfg.read_text())!=config:raise ValueError('Run prepare with this exact sampling config')
    entries=json.loads(manifest.read_text())['episodes']
    if a.full:
        if (len(entries),sum(e['length'] for e in entries))!=(30000,27815381):raise ValueError('Full scope differs')
    elif not 1<=len(entries)<=8:
        raise ValueError('Bounded mode permits only 1..8 episodes; full requires explicit --full')
    if a.stage=='generate':
        subprocess.run([sys.executable,'-m','starVLA.pri4r.generate','--manifest',str(manifest),
            '--config',str(cfg),'--output',str(gt),'--workers',str(a.workers),
            '--shard-index',str(a.shard_index),'--num-shards',str(a.num_shards)],
            env=environment('pri4r_paper'),check=True)
    elif a.stage=='verify':
        activate('pri4r_paper')
        from starVLA.pri4r.reader import episode_key,validate_episode,PointTrackStore
        seen=set();frames=0
        for e in entries:
            key=episode_key(e['source_prefix'],e['source_episode_index'])
            if key in seen:raise ValueError('Duplicate source episode')
            seen.add(key);meta,arrays=validate_episode(gt/key,verify_hashes=True)
            if meta['config_sha256']!=digest or meta['frames']!=e['length'] or meta['points']!=1024:
                raise ValueError('GT scope/config differs: '+key)
            if meta['generator']!='rigid_surface_replay_v1':raise ValueError('Wrong generator')
            proof=meta['alignment_evidence']
            if proof.get('passed') is not True:raise ValueError('Source/current alignment missing')
            for k in ('source_prefix','source_episode_index','current_dataset_name','current_episode_index','length'):
                if proof[k]!=e[k]:raise ValueError('Episode mapping differs: '+k)
            PointTrackStore._close_arrays(arrays);frames+=meta['frames']
        proof={'method':'pri4r_paper','bounded':not a.full,'episodes':len(seen),'frames':frames,
               'config_sha256':digest,'manifest_sha256':sha256(manifest),'verified_all_array_hashes':True}
        (gt/'PORTABLE_GT_ACCEPTED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':main()
