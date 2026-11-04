from pathlib import Path
import json,re
from contract import sha256

def compatible_checkpoint(path,fingerprint,world,verify_hashes=True):
 path=Path(path);marker=path/'COMPLETE.json'
 if not marker.is_file():return None
 m=json.loads(marker.read_text())
 if m.get('fingerprint')!=fingerprint or m.get('world_size')!=world:return None
 assert m['complete'] and sorted(m['rank_rng'])==list(range(world))
 assert m['step']==int(path.name.split('_')[1])
 assert set(m['files'])=={'training.pt',*[f'rng_rank{r}.pt' for r in range(world)]}
 for name,row in m['files'].items():
  f=path/name
  assert f.is_file() and f.stat().st_size==row['bytes'],str(f)
  if verify_hashes:assert sha256(f)==row['sha256'],str(f)
 return m

def latest_checkpoint(root,fingerprint,world):
 rejected=[]
 for p in sorted(Path(root).glob('step_*'),reverse=True):
  if not re.fullmatch(r'step_\d{6}',p.name):continue
  try:
   m=compatible_checkpoint(p,fingerprint,world)
   if m:return p,m,rejected
   rejected.append((p.name,'missing complete marker or incompatible recipe/world'))
  except (AssertionError,ValueError,OSError) as e:rejected.append((p.name,str(e)))
 return None,None,rejected
