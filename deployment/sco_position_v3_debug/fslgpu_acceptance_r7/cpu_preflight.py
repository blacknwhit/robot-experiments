from pathlib import Path
import os,subprocess,time
from controller import HERE,CASES,verify,command
from gdb_runner import prerequisites
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
verify();out=HERE/('cpu_'+str(time.time_ns()));out.mkdir()
for case in CASES:
    cmd=command(case,out/case[0])
    for flag in ('--config','--checkpoint'):assert Path(cmd[cmd.index(flag)+1]).is_file(),flag
    subprocess.run(cmd+['--dry-run'],check=True)
prerequisites()
