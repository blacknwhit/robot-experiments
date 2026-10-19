"""Download public inputs only. Large downloads require an explicit component."""
import argparse, json, shutil, urllib.parse, urllib.request
from pathlib import Path
from common import ROOT, sha256
DATASETS={'atomic':('ember-lab-berkeley/robocasa365-pretrain-atomic','d5eac857ee9b07ff089e89c594c64493b8df6e41'),
          'composite':('ember-lab-berkeley/robocasa365-pretrain-composite','4a5e4552f8823b8c92944e08bba1ff268c51bef9')}
NORM_SHA='4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
def fetch(url,path,expected=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=180) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
        if expected and sha256(tmp)!=expected:raise ValueError(f'Digest mismatch: {path}')
        tmp.replace(path)
    if expected and sha256(path)!=expected:raise ValueError(f'Digest mismatch: {path}')
    return {'file':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'url':url}
def main():
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);p.add_argument('--component',choices=('small','datasets','base'),required=True);a=p.parse_args()
    h=a.home.resolve();receipts=[]
    if a.component=='small':
        receipts.append(fetch('https://huggingface.co/robocasa/robocasa365_checkpoints/resolve/916d8ec4aab4e6f930ba6b5b9bb8537ad16d4bb8/pi05_pretrain_human300/multitask_learning/75000/assets/norm_stats.json',h/'models/robocasa_norm/norm_stats.json',NORM_SHA))
        receipts.append(fetch('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',h/'cache/openpi/big_vision/paligemma_tokenizer.model'))
    elif a.component=='datasets':
        from huggingface_hub import snapshot_download
        for repo,rev in DATASETS.values():
            target=h/'data/robocasa365'/repo.split('/')[-1]
            snapshot_download(repo_id=repo,repo_type='dataset',revision=rev,local_dir=target)
            receipts.append({'repo':repo,'revision':rev,'path':str(target)})
    else:
        # Public GCS JSON listing, no gsutil credentials required. Pin each object generation.
        prefix='checkpoints/pi05_base/';token=None;objects=[]
        while True:
            params={'prefix':prefix,**({'pageToken':token} if token else {})}
            with urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?'+urllib.parse.urlencode(params),timeout=60) as f:d=json.load(f)
            objects+=d.get('items',[]);token=d.get('nextPageToken')
            if not token:break
        if not any(o['name'].endswith('/commit_success.txt') for o in objects):raise ValueError('Missing Orbax completion object')
        for o in objects:
            if o['name'].endswith('/'):continue
            path=h/'models/pi05_base'/o['name'][len(prefix):]
            url='https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/'+urllib.parse.quote(o['name'],safe='')+'?alt=media&generation='+o['generation']
            rec=fetch(url,path)
            if rec['bytes']!=int(o['size']):raise ValueError('GCS size mismatch')
            rec['generation']=o['generation'];receipts.append(rec)
    out=h/'receipts'/f'download_{a.component}.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipts,indent=2));print(out)
if __name__=='__main__':main()
