"""Rebuild the exact selected 300-task/100-demo scope using historical sampler functions."""
import argparse,ast,json,random
from pathlib import Path
from common import ROOT,runtime,sha256
def main():
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);a=p.parse_args();h=a.home.resolve()
    import pandas as pd
    source=runtime('query3d_v2')/'examples/pi05_robocasa365/train_files/data_registry/data_config.py'
    names={'_load_pretrain300_tasks','_task_from_source_prefix','_filter_task_episodes'}
    tree=ast.parse(source.read_text());module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    ns={'ast':ast,'Path':Path,'pd':pd,'random':random};exec(compile(module,str(source),'exec'),ns)
    tasks=ns['_load_pretrain300_tasks'](ROOT/'assets/dataset_registry.py');rows=[];metadata={}
    for dataset in ['robocasa365-pretrain-atomic','robocasa365-pretrain-composite']:
        root=h/'data/robocasa365'/dataset;files=sorted((root/'meta/episodes').rglob('*.parquet'))
        if not files:raise FileNotFoundError(root/'meta/episodes')
        metadata.update({dataset+'/'+f.relative_to(root).as_posix():sha256(f) for f in files})
        df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
        columns=[c for c in df.columns if c in ['episode_index','length','source_prefix','source_episode_index','dataset_from_index','dataset_to_index'] or c.startswith('data/')]
        df=df[columns];df['dataset_name']=dataset;df['task_name']=df.source_prefix.map(ns['_task_from_source_prefix']);rows.append(df)
    df=pd.concat(rows,ignore_index=True);selected=[];counts={}
    for task in tasks:
        sub=df[df.task_name==task]
        if len(set(sub.dataset_name))!=1:raise ValueError('Task partition missing/ambiguous: '+task)
        chosen=ns['_filter_task_episodes'](sub,'100_demos',0)
        if len(chosen)!=100:raise ValueError('Expected 100 demos: '+task)
        counts[task]={'selected':len(chosen),'frames':int(chosen.length.sum())};selected+=chosen.to_dict(orient='records')
    result={'schema':'query_pri4r_training_scope_audit_v1','filter_key':'100_demos','filter_seed':0,'weight_alpha':0.4,
        'tasks':len(tasks),'episodes':len(selected),'frames':sum(int(x['length']) for x in selected),'counts':counts,
        'selected_episodes':selected,'registry_sha256':sha256(ROOT/'assets/dataset_registry.py'),'metadata_sha256':metadata}
    if (result['tasks'],result['episodes'],result['frames'])!=(300,30000,27815381):raise ValueError('Historical training scope differs')
    out=h/'data/training_scope.json';out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(result,indent=2)
    if out.exists() and out.read_text()!=text:raise ValueError('Existing scope changed')
    out.write_text(text);print(out)
if __name__=='__main__':main()
