"""Fetch pinned teacher code/weights into an explicit resource directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = {
    'vggt': ('facebookresearch/vggt','a288dd0f14786c93483e45524328726ab7b1b4ce'),
    'Track4World': ('TencentARC/Track4World','1c2a0f814d8352d2750527b70a6c546e056c2c67'),
    'utils3d': ('jiah-cloud/utils3d','2072c024c73f7c0f83e0da23eef5f2d9ac575249'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); args=p.parse_args()
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    for name,(repo,commit) in SOURCES.items():
        dest=root/'upstream'/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','https://github.com/'+repo+'.git',str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',commit],check=True)
        head=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        assert head==commit,(name,head,commit)
    # DA3-only extraction does not instantiate Pi3/Pi3X. Avoid importing their
    # optional code/extensions; the selected DA3 forward/weights are unchanged.
    file=root/'upstream/Track4World/track4world/nets/model.py'
    code=file.read_text(encoding='utf-8')
    for line in ('from track4world.nets.external.pi3.models.pi3 import Pi3\n',
                 'from track4world.nets.external.pi3x.models.pi3x import Pi3X\n'):
        code=code.replace(line,'')
    # The released checkpoint restores all ConvNeXt keys. Skip an otherwise
    # redundant torchvision weight download during teacher construction.
    code=code.replace('ConvNeXt(block_setting, stochastic_depth_prob=0.0, init_weights=True)',
                      'ConvNeXt(block_setting, stochastic_depth_prob=0.0, init_weights=False)')
    file.write_text(code,encoding='utf-8')
    from huggingface_hub import hf_hub_download
    manifest=json.loads((Path(__file__).resolve().parents[2]/'runtimes/focalvla/teachers.json').read_text())
    for name,entry in manifest.items():
        for item in entry['files']:
            file=Path(hf_hub_download(entry['repo'],item['name'],revision=entry['revision'],local_dir=root/'weights'/name))
            if item['sha256']:
                h=hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
                assert h.hexdigest()==item['sha256'],file
    fragment={'vggt_source':str(root/'upstream/vggt'),'track4world_source':str(root/'upstream/Track4World'),
              'utils3d_source_parent':str(root/'upstream'),'vggt_weights':str(root/'weights/vggt/model.pt'),
              'track4world_weights':str(root/'weights/track4world/track4world_da3.pth'),'da3_weights':str(root/'weights/da3')}
    (root/'teacher_resources.json').write_text(json.dumps(fragment,indent=2)+'\n')
    print(json.dumps(fragment,indent=2))


if __name__=='__main__': main()
