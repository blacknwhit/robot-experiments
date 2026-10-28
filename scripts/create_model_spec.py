"""Create an editable model descriptor without changing benchmark source code."""
import argparse
import json
from pathlib import Path
import sys
from common import VARIANTS,runtime,import_paths
from robot_eval.spec import FORMAT,load_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--variant',choices=VARIANTS,help='Use an existing native PI0.5 runtime')
    p.add_argument('--factory',default='pi05',help='pi05 or model_package.adapter:create')
    p.add_argument('--source-root',type=Path,action='append',default=[])
    p.add_argument('--python-path',type=Path,action='append',default=[])
    p.add_argument('--asset',action='append',default=[],help='NAME=/absolute/path to normalization/tokenizer/base assets')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--policy-python',type=Path,default=Path(sys.executable))
    p.add_argument('--native',action='store_true',help='Use model-native prediction instead of PI0.5 CUDA Graph')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Descriptor exists; use a new path or edit it explicitly')
    roots=a.source_root;paths=a.python_path
    if a.variant:
        if roots or paths or a.factory!='pi05':p.error('--variant selects its own source and factory')
        roots=[runtime(a.variant)];paths=import_paths(a.variant)
    if not roots or not paths:p.error('Specify --variant or model-owned --source-root and --python-path')
    assets={}
    for item in a.asset:
        key,sep,value=item.partition('=')
        if not sep or not key or key in assets:p.error('Assets must have unique NAME=PATH')
        assets[key]=str(Path(value).resolve())
    data={'format':FORMAT,'model_id':a.model_id,'factory':a.factory,
          'config':str(a.config.resolve()),'checkpoint':str(a.checkpoint.resolve()),
          'policy_python':str(a.policy_python.resolve()),'python_paths':[str(p.resolve()) for p in paths],
          'source_roots':[{'path':str(p.resolve()),'include':['**/*.py']} for p in roots],
          'assets':assets,'options':{},'environment':{}}
    if a.native:data.update(backend='native',backend_version='model_native_v1',max_concurrency=1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    load_spec(a.output)
    print(str(a.output.resolve()))


if __name__=='__main__':main()
