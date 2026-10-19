import argparse,subprocess,sys
from pathlib import Path
from common import environment
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);a=p.parse_args();h=a.home.resolve()
    subprocess.run([sys.executable,'-m','starVLA.dataloader.gr00t_lerobot.prepare_v3_arrow_cache','--cache-dir',str(h/'cache/arrow'),
        str(h/'data/robocasa365/robocasa365-pretrain-atomic'),str(h/'data/robocasa365/robocasa365-pretrain-composite')],env=environment('query3d_v2'),check=True)
