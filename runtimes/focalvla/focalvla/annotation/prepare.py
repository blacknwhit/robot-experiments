"""Portable scene replay and task-generic offline annotation."""
from pathlib import Path
import os,json,gzip,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .templates import allowed_templates
from .grounding import entity_geom_ids,motion_hints,body_membership
from ..common import atomic_json as save
from ..simulation import rewrite_assets,restore_state
RESOURCES=json.loads(Path(os.environ['FOCAL_RESOURCES']).read_text())
D=Path(os.environ['FOCAL_PREPARED'])/'annotations'
SOURCE=Path(RESOURCES['source_root'])

def entities(meta,instruction):
    result={}
    for c in meta.get('object_cfgs',[]):
        n=c['name']
        if n.startswith('distr'):continue
        result[n]={'name':n,'description':c.get('info',{}).get('cat',str(c.get('obj_groups',n))),
                   'kind':'object','aliases':[]}
    fixtures=meta.get('fixtures',{})
    def add_fixture(n,alias):
        if not isinstance(n,str) or n not in fixtures:return
        result.setdefault(n,{'name':n,'description':str(fixtures[n].get('cls',n)),
                             'kind':'fixture','aliases':[]})['aliases'].append(alias)
    for alias,n in meta.get('fixture_refs',{}).items():
        if isinstance(n,dict):n=n.get('name')
        if isinstance(n,list):
            for item in n:add_fixture(item,alias)
        else:add_fixture(n,alias)
    for c in meta.get('object_cfgs',[]):
        if not c['name'].startswith('distr'):add_fixture(c.get('placement',{}).get('fixture'),'source support')
    keywords={'stove':['stove','burner'],'sink':['sink','faucet','spout'],'fridge':['fridge','freezer'],
      'microwave':['microwave'],'toaster':['toaster'],'oven':['oven'],'blender':['blender'],
      'kettle':['kettle'],'coffeemachine':['coffee machine','coffee maker'],'dishwasher':['dishwasher'],
      'standmixer':['stand mixer'],'drawer':['drawer'],'cabinet':['cabinet']}
    for kind,words in keywords.items():
        if any(w in instruction.lower() for w in words):
            # Explicit scene references already resolve the relevant fixture.
            # Do not add every cabinet/drawer in the kitchen as another candidate.
            matching=lambda c:kind in str(c).lower().replace('_','').replace(' ','')
            if any(e['kind']=='fixture' and matching(e['description']) for e in result.values()):continue
            if kind=='drawer' and 'fridge' in instruction.lower() and any('fridge' in e['description'].lower() for e in result.values()):continue
            for n,c in fixtures.items():
                if matching(c.get('cls','')):add_fixture(n,'instruction target')
    return [dict(item,id=i) for i,item in enumerate(result.values())]

def make_sim(ep):
    xml,_=rewrite_assets(gzip.decompress((ep/'model.xml.gz').read_bytes()),RESOURCES['asset_roots'])
    meta=json.loads((ep/'ep_meta.json').read_text());tree=ET.fromstring(xml)
    for name,cfg in meta['cam_configs'].items():
        parent=tree.find(f".//body[@name='{cfg['parent_body']}']") if cfg.get('parent_body') else tree.find('worldbody')
        if parent is None:continue
        cam=parent.find(f".//camera[@name='{name}']")
        if cam is None:cam=ET.SubElement(parent,'camera',name=name,mode='fixed')
        if 'sensorsize' in cfg.get('camera_attribs',{}):cam.attrib.pop('fovy',None)
        cam.set('pos',' '.join(map(str,cfg['pos'])));cam.set('quat',' '.join(map(str,cfg['quat'])));cam.attrib.update(cfg.get('camera_attribs',{}))
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));model.vis.quality.offsamples=0
    return model,mujoco.MjData(model),np.load(ep/'states.npz')['states'],meta

def prepare_case(row):
    name=f"{row['task_name']}__{row['source_episode_index']:06d}";dest=D/'cases'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'READY.json').exists():
        cached=json.loads((dest/'READY.json').read_text())
        if cached.get('version')==3:return cached
    ep=SOURCE/row['source_prefix']/'lerobot/extras'/f"episode_{row['source_episode_index']:06d}"
    model,data,states,meta=make_sim(ep)
    instruction=row['tasks'][0];es=entities(meta,instruction)
    if 'Lid' in row['task_name'] and not any('counter' in e['description'].lower() for e in es):
        # A removable lid is often put onto a nearby counter even when that
        # support is not explicitly listed among the task's fixture references.
        counters=[{'name':n,'description':c['cls'],'kind':'fixture','aliases':['nearby lid support'],'id':0}
                  for n,c in meta.get('fixtures',{}).items() if c.get('cls')=='Counter']
        restore_state(model,data,states[0]);mujoco.mj_kinematics(model,data)
        anchor=np.mean([data.xpos[m[0]] for m in body_membership(model,es)],axis=0)
        ranked=sorted(zip(counters,body_membership(model,counters)),key=lambda pair:np.linalg.norm(data.xpos[pair[1][0]][:2]-anchor[:2]))
        for e,_ in ranked[:2]:e['id']=len(es);es.append(e)
    gids=entity_geom_ids(model,es)
    hints=motion_hints(model,states,es)
    # Unknown entities are retained as a detectable mapping failure, never substituted.
    assert es and all(gids),[(e['name'],len(g)) for e,g in zip(es,gids)]
    times=[];pixel_counts=[];cams=['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']
    info={'name':name,'task':row['task_name'],'instruction':instruction,'length':row['length'],
        'source_prefix':row['source_prefix'],'source_episode_index':row['source_episode_index'],
        'episode_index':row['episode_index'],'dataset_name':row['dataset_name'],'frames':times,'entities':es,
        'visible_pixels':pixel_counts,'allowed_templates':allowed_templates(row['task_name']),
        'layout_id':meta.get('layout_id'),'style_id':meta.get('style_id'),'cameras':cams,'version':3,'mask_renderer':'OSMesa,offSamples0',
        'entity_mapping':'complete body subtree including anonymous descendants; exact root prevents prefix collisions',
        'motion_hints':hints}
    save(dest/'READY.json',info);print('PREPARED',name,'entities',len(es),flush=True)
    return info

