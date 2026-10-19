"""Convert the clean official PI0.5 Orbax base, never the RoboCasa75k policy."""
import argparse,json,subprocess,sys
from pathlib import Path
from common import runtime,environment,sha256
def main():
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);a=p.parse_args();h=a.home.resolve()
    source=h/'models/pi05_base';target=h/'models/pi05_base_pytorch'
    if not (source/'params/commit_success.txt').is_file():raise FileNotFoundError('Download the complete clean base first')
    if target.exists():raise FileExistsError('Conversion output already exists; inspect it, do not overwrite')
    r=runtime('baseline');env=environment('baseline');env['OPENPI_DATA_HOME']=str(h/'cache/openpi')
    subprocess.run([sys.executable,str(r/'playground/Code/openpi_robocasa365/examples/convert_jax_model_to_pytorch.py'),
        '--checkpoint-dir',str(source),'--config-name','pi05_pretrain_human300','--output-path',str(target),'--precision','bfloat16'],env=env,check=True)
    weights=target/'model.safetensors';digest=sha256(weights)
    rec={'source':'gs://openpi-assets/checkpoints/pi05_base','sha256':digest,'bytes':weights.stat().st_size,
         'historical_sha256':'9fd36e337832960bbfc122bd686dfc64b89551522d891c0d21ae586385de297e'}
    (target/'conversion_manifest.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
    if digest!=rec['historical_sha256']:raise RuntimeError('Conversion bytes differ from historical base: inspect tensors/provenance before training')
if __name__=='__main__':main()
