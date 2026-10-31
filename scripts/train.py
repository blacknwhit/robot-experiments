"""Launch the selected historical trainer with explicit state ownership."""
import argparse, json, os, subprocess, sys
from pathlib import Path
import yaml
from common import ROOT, environment, runtime, POSITION_VARIANTS
from release_contract import verify_release

def training_environment(config):
    variant=config['portable']['variant'];env=environment(variant)
    if variant in POSITION_VARIANTS or variant == 'pri4r_paper':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    resource_home = config['portable'].get('resource_home', Path(config['run_root_dir']).parent)
    env['OPENPI_DATA_HOME']=str(Path(resource_home)/'cache/openpi')
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c=yaml.safe_load(a.config.read_text());v=c['portable']['variant'];n=c['portable']['world_size']
    verify_release(c)
    run=Path(c['run_root_dir'])/c['run_id'];env=training_environment(c)
    batch = 128 if v == 'pri4r_paper' else 64
    acc = c['trainer']['gradient_accumulation_steps']
    micro = c['datasets']['vla_data']['per_device_batch_size']
    if micro*n*acc != batch:raise ValueError('Global batch mismatch')
    ds={'fp16':{'enabled':False},'bf16':{'enabled':True},'train_micro_batch_size_per_gpu':micro,'train_batch_size':batch,
        'gradient_accumulation_steps':acc,'zero_optimization':{'stage':2,'allgather_partitions':True,'allgather_bucket_size':500000000,
        'reduce_scatter':True,'reduce_bucket_size':500000000,'overlap_comm':True,'contiguous_gradients':True,'cpu_offload':False},'gradient_clipping':1.0,'steps_per_print':10}
    control=run.parent/'.launch'/c['run_id'];ds_path=control/'deepspeed.json';acc_path=control/'accelerate.yaml'
    accelerator_config={'compute_environment':'LOCAL_MACHINE','distributed_type':'DEEPSPEED','num_machines':1,'num_processes':n,
         'deepspeed_config':{'deepspeed_config_file':str(ds_path),'deepspeed_multinode_launcher':'standard','zero3_init_flag':False}}
    cmd=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(acc_path),str(ROOT/'scripts/train_entry.py'),'--config',str(a.config.resolve())]
    if a.resume:cmd+=['--resume']
    print(json.dumps({'command':cmd,'run':str(run),'world':n,'micro_batch':micro,'global_batch':batch,'accumulation':acc,'resume':a.resume},indent=2))
    if a.dry_run:return
    control.mkdir(parents=True,exist_ok=True);ds_path.write_text(json.dumps(ds,indent=2));acc_path.write_text(yaml.safe_dump(accelerator_config))
    # This process retains the lock while torchrun and every worker are alive.
    import fcntl
    with (control/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run(cmd,cwd=runtime(v),env=env,check=True)
if __name__=='__main__':main()
