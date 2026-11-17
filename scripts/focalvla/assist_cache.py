"""Validate and merge isolated assistant outputs after primary writers exit."""
import json,hashlib,os,time,shutil
from pathlib import Path
import numpy as np

NAMES={'state','action','frames','geo','motion','geo_valid','motion_valid','goal_ids'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_bytes(data);tmp.replace(p)
def save(p,x):atomic(p,(json.dumps(x,indent=2)+'\n').encode())
def label_path(root,row):return Path(root)/'annotations/physics_v5'/row['name']/'LABELS.json'
def complete(root,row,frames,identity):
    cache=Path(root)/'cache'/f"{row['id']:06d}";marker=cache/'COMPLETE.json'
    if not marker.exists():return None
    m=read(marker)
    if m['identity']!=identity:raise ValueError(f'Wrong cache identity: {marker}')
    if m.get('image_storage')!='source_video':raise ValueError(f'Wrong storage: {marker}')
    if set(m['file_bytes'])!={n+'.npy' for n in NAMES}:raise ValueError(f'Unexpected cache files: {marker}')
    if not all((cache/f).exists() and (cache/f).stat().st_size==n for f,n in m['file_bytes'].items()):return None
    lab=label_path(root,row)
    if not lab.exists() or sha(lab)!=m['label_sha256']:return None
    expected=digest({'inputs':identity,'label_sha256':sha(lab),'episode':row,'frames':list(map(int,frames))})
    if m['contract']!=expected:raise ValueError(f'Changed episode contract: {marker}')
    if m['frames']!=len(frames) or not np.array_equal(np.load(cache/'frames.npy'),frames):raise ValueError('Frame mismatch')
    return m
def coverage(primary,helper,rows,records,identity):
    sources=[];targets=0
    for row in rows:
        lo,hi=np.searchsorted(records[:,0],[row['id'],row['id']+1]);frames=records[lo:hi,1]
        source='primary' if complete(primary,row,frames,identity) else ('helper' if complete(helper,row,frames,identity) else None)
        if source is None:return None
        sources.append([row['id'],source]);targets+=len(frames)
    return {'episodes':len(rows),'targets':targets,'sources':sources,'sources_digest':digest(sources),'identity_digest':digest(identity)}
def merge(primary,helper,rows,records,identity,*,writers_stopped=False):
    if not writers_stopped:raise RuntimeError('Primary writer must be stopped before merge')
    primary=Path(primary).resolve();helper=Path(helper).resolve()
    if primary==helper or primary.is_relative_to(helper) or helper.is_relative_to(primary):raise ValueError('Cache roots overlap')
    if coverage(primary,helper,rows,records,identity) is None:raise RuntimeError('Incomplete union')
    result={'kept_primary':0,'merged_helper':0}
    for row in rows:
        lo,hi=np.searchsorted(records[:,0],[row['id'],row['id']+1]);frames=records[lo:hi,1]
        if complete(primary,row,frames,identity):result['kept_primary']+=1;continue
        source=helper/'cache'/f"{row['id']:06d}";dest=primary/'cache'/f"{row['id']:06d}"
        assert source.resolve().is_relative_to(helper) and dest.resolve().is_relative_to(primary)
        lab=label_path(primary,row);other=label_path(helper,row)
        if lab.exists() and read(lab)!=read(other):raise ValueError(f'Conflicting primary annotation: {row["name"]}')
        atomic(lab,other.read_bytes())
        # Preserve partial old writes for audit. Never overwrite a complete primary episode.
        if dest.exists():
            archive=primary/'assist_partial_archive'/f"{row['id']:06d}_{time.time_ns()}"
            archive.parent.mkdir(exist_ok=True)
            assert archive.resolve().is_relative_to(primary)
            dest.rename(archive)
        dest.parent.mkdir(parents=True,exist_ok=True)
        source.rename(dest)
        assert complete(primary,row,frames,identity)
        result['merged_helper']+=1
    return result
