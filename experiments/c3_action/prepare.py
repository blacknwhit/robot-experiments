"""Verify existing GT or resolve a new run. Does not regenerate tracks or train."""
from pathlib import Path
import argparse,json,sys
import yaml
from contract import ROOT,activate,digest,sha256,write

def verify_gt(gt):
 activate()
 from starVLA.pri4r.reader import validate_episode,PointTrackStore
 config=json.loads((gt/'sampling_config.json').read_text());h=digest(config)
 expected=json.loads((ROOT/'configs/gt_sampling.json').read_text())
 assert {k:v for k,v in config.items() if k!='asset_root'}=={k:v for k,v in expected.items() if k!='asset_root'},'C3 requires original arm/object sampling, not paper cube GT'
 count=frames=0;metadata={}
 for marker in sorted(gt.rglob('complete.json')):
  m,arrays=validate_episode(marker.parent,verify_hashes=True)
  assert m['config_sha256']==h and m['points']==1024 and m['fps']==20
  assert m['revision']=='522e4ffa3bb9d9729f79b241c65fc9e589f2659c' and m['repo']=='nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos'
  assert m['state_to_frame']=='same_index_no_integration' and m['alignment_evidence']['passed']
  assert set(m['clock_diagnostic']['non_nominal_interval_indices'])<={0},'Unexpected internal simulator clock gap'
  assert m['episode_key']==marker.parent.relative_to(gt).as_posix()
  metadata[m['episode_key']]=sha256(marker);count+=1;frames+=m['frames'];PointTrackStore._close_arrays(arrays)
  if count%500==0:print('GT_VERIFIED',count,flush=True)
 assert (count,frames)==(30000,27815381),(count,frames)
 write(gt/'C3_GT_ACCEPTED.json',{'schema':'c3_reused_armobj_gt_v1','episodes':count,'frames':frames,
       'config_sha256':h,'all_array_hashes_verified':True,'episode_metadata':metadata,
       'clock_handling':'mask whole auxiliary horizon when source frame_index is zero'})

def resolve(home,gt,run_root,run_id,world,mode,arm,schedule):
 import re
 if re.fullmatch('[A-Za-z0-9_.-]+',run_id) is None:raise ValueError('Simple run-id required')
 if world not in (1,2,4,8) or (mode=='formal' and world!=8):raise ValueError('Formal=8 GPUs; smoke=1/2/4/8')
 proof=json.loads((gt/'C3_GT_ACCEPTED.json').read_text())
 text=(ROOT/'configs/embedding3d_v2.yaml').read_text();runtime=ROOT/'runtimes/trajectory_v2'
 replacements={'OPENPI_ROOT':runtime/'playground/Code/openpi_robocasa365','BASE_WEIGHTS':home/'models/pi05_base_pytorch/model.safetensors',
  'NORM_ASSETS':home/'models/robocasa_norm','DATA_ROOT':home/'data/robocasa365','REGISTRY_FILE':ROOT/'assets/dataset_registry.py',
  'ARROW_CACHE':home/'cache/arrow','GT_ROOT':gt,'GT_CONFIG_SHA256':proof['config_sha256'],'RUN_ID':run_id,'RUN_ROOT':run_root}
 for k,v in replacements.items():text=text.replace('${'+k+'}',str(v))
 assert '${' not in text
 framework=yaml.safe_load(text)
 framework['datasets']['vla_data'].update(per_device_batch_size=8,num_workers=4,prefetch_factor=2)
 # The wrapper applies the calibrated objective and its own 100-step warmup.
 framework['framework']['pi05']['point_track']['auxiliary_warmup_steps']=0
 return {'schema':'c3_action_full70k_v1','arm':arm,'home':str(home),'gt_root':str(gt),'run':str(run_root/run_id),
  'mode':mode,'world':world,'micro_batch':8,'accumulation':64//world//8,'global_batch':64,'seed':42,
  'steps':70000,'schedule_steps':schedule,'warmup_steps':1000,'learning_rate':2.5e-5,'min_learning_rate':2.5e-6,
  'head_learning_rate':1e-4,'aux_lambda':0.07516936569041761,'aux_warmup_steps':100,
  'clip_main':1.,'clip_aux':1.,'optimizer':{'betas':[.9,.95],'eps':1e-8,'weight_decay':1e-10},
  'save_interval':5000,'data_workers':4,'prefetch_factor':2,'sampling':'baseline_task_weighted_ordered_group8_v1',
  'gt_config_sha256':proof['config_sha256'],'gt_receipt_sha256':sha256(gt/'C3_GT_ACCEPTED.json'),
  'base_sha256':sha256(home/'models/pi05_base_pytorch/model.safetensors'),
  'norm_sha256':sha256(home/'models/robocasa_norm/norm_stats.json'),
  'source_manifest_sha256':sha256(ROOT/'provenance/source_files.sha256'),'framework_config':framework}

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 v=sub.add_parser('verify-gt');v.add_argument('--gt-root',type=Path,required=True)
 r=sub.add_parser('configure');r.add_argument('--home',type=Path,required=True);r.add_argument('--gt-root',type=Path)
 r.add_argument('--run-root',type=Path,required=True);r.add_argument('--run-id',required=True)
 r.add_argument('--gpus',type=int,default=8);r.add_argument('--mode',choices=['smoke','formal'],required=True)
 r.add_argument('--arm',choices=['C0','C1','C3'],default='C3');r.add_argument('--schedule-steps',type=int,choices=[70000,100000],default=100000)
 r.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.action=='verify-gt':verify_gt(a.gt_root.resolve());return
 assert not a.out.exists(),'Do not overwrite a frozen configuration'
 home=a.home.resolve();gt=(a.gt_root or home/'data/gt_armobj_full_v1').resolve()
 c=resolve(home,gt,a.run_root.resolve(),a.run_id,a.gpus,a.mode,a.arm,a.schedule_steps)
 write(a.out,c);print(a.out.resolve())
if __name__=='__main__':main()
