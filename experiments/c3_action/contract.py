"""Portable C3 recipe; separate from all historical runtime/train entry points."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'scripts'
sys.path.insert(0,str(SCRIPTS))
from common import environment as base_environment,activate as base_activate,sha256

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,p)
def activate():return base_activate('embedding3d_v2')
def env(c):
 e=base_environment('embedding3d_v2');home=Path(c['home'])
 e.update(OPENPI_DATA_HOME=str(home/'cache/openpi'),PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8',
          TORCH_COMPILE_DISABLE='1',NVIDIA_TF32_OVERRIDE='0',WANDB_MODE='disabled')
 e['PYTHONPATH']=str(ROOT/'experiments/c3_action')+os.pathsep+str(SCRIPTS)+os.pathsep+e['PYTHONPATH']
 return e
def fingerprint(c):return digest(c)
def validate(c):
 assert c['schema']=='c3_action_full70k_v1' and c['arm'] in ('C0','C1','C3')
 assert c['world'] in (1,2,4,8) and c['micro_batch']==8 and c['global_batch']==64
 assert c['accumulation']*c['micro_batch']*c['world']==64
 assert c['mode'] in ('smoke','formal')
 if c['mode']=='formal':assert c['world']==8 and c['accumulation']==1
 assert c['steps']==70000 and c['warmup_steps']==1000 and c['schedule_steps'] in (70000,100000)
 assert c['aux_warmup_steps']==100 and c['aux_lambda']==0.07516936569041761
 assert c['sampling']=='baseline_task_weighted_ordered_group8_v1'
 assert c['source_manifest_sha256']==sha256(ROOT/'provenance/source_files.sha256')
 from release_contract import verify_release
 verify_release({'trajectory_release_sha256':c['source_manifest_sha256']})
 gt=Path(c['gt_root']);proof=json.loads((gt/'C3_GT_ACCEPTED.json').read_text())
 assert sha256(gt/'C3_GT_ACCEPTED.json')==c['gt_receipt_sha256']
 assert proof['config_sha256']==c['gt_config_sha256'] and proof['episodes']==30000
 assert proof['frames']==27815381 and proof['all_array_hashes_verified'] is True
 assert sha256(Path(c['home'])/'models/robocasa_norm/norm_stats.json')==c['norm_sha256']
 return proof
