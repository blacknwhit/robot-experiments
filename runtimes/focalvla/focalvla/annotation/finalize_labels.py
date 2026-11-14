"""Complete reviewed coarse goals without episode-specific frame constants."""
from pathlib import Path
import sys,json,re,copy
from .prepare import D
from .physics_labels_v4 import Replay,natural,mujoco,restore_state,np,save

def event_list(lab):
    return [e for g in lab['goals'] for e in ([g['event']] if 'event' in g else [])+g.get('additional_events',[])]

def finish(name):
    lab=json.loads((D/'physics_v4'/name/'LABELS.json').read_text());changes=[];task=lab['task']
    ready=json.loads((D/'cases'/name/'READY.json').read_text());r=None
    # Preserve explicitly named target/reference objects in an atomic goal.
    # E.g. cheese rests on bread, whose category is bread_flat, above a board.
    if '/atomic/' in ready['source_prefix']:
        clauses=re.findall(r'\b(?:on|onto|in|into|inside|contains|containing|under|beside|next to)\b([^.;]+)',lab['instruction'].lower())
        for e in ready['entities']:
            if e['kind']!='object':continue
            words=[w for w in natural(e).split() if len(w)>=4 and w not in ['flat','electric','non']]
            if any(re.search(r'\b'+re.escape(w)+r's?\b',clause) for w in words for clause in clauses):
                for goal in lab['goals']:
                    if e['id'] not in goal['entities']:
                        goal['entities']=sorted(goal['entities']+[e['id']]);changes.append(f'Preserve explicitly named atomic target/reference entity {e["id"]}.')
    if task!='RinseCuttingBoard' and any(g['kind']=='rinse' for g in lab['goals']):
        r=Replay(D/'cases'/name);events=event_list(lab)
        for gi,goal in enumerate(lab['goals']):
            if goal['kind']!='rinse':continue
            ev=goal['event'];src=ev['source'];dst=ev['target'];actor=ev['primary'];sink_ids=[e['id'] for e in r.es if natural(e)=='sink']
            was_placed=any(e['core_start']<ev['core_start'] and
                ((e.get('primary')==actor and e.get('target') in sink_ids) or
                 any(x['entity']==actor and x['target'] in sink_ids for x in e.get('transferred_contents',[]))) for e in events)
            water=[]
            for sink in sink_ids:
                for joint in r.joints[sink]:
                    joint_name=mujoco.mj_id2name(r.m,mujoco.mjtObj.mjOBJ_JOINT,joint) or ''
                    if joint_name.endswith('_handle_joint'):
                        q=r.qpos[ev['core_start']:ev['core_end']+1,int(r.m.jnt_qposadr[joint])];water.extend((abs(q)>.08).tolist())
            dry_placement=dst in sink_ids and water and not any(water)
            washing_finished=src in sink_ids and dst not in sink_ids and was_placed
            if dry_placement or washing_finished:
                noun=lambda i:natural(r.es[i]) if i>=0 else 'the demonstrated location'
                goal['kind']='transfer';goal['text']=f'Move the {noun(actor)} from the {noun(src)} to the {noun(dst)}.'
                changes.append('Distinguish sink placement or retrieval from the intervening rinse stage.')
        # When food is rinsed while stationary in the basin, retain that food
        # in the relevance mask during the intervening faucet/waiting goal.
        involved={e['primary'] for e in events if e['kind']=='grasp'}|{x['entity'] for e in events for x in e.get('transferred_contents',[])}
        for goal in lab['goals']:
            ev=goal.get('event',{});sink=ev.get('primary')
            if ev.get('kind')!='control' or sink not in sink_ids:continue
            stationary=[]
            for actor in involved:
                if actor==sink or actor not in r.frees:continue
                support,_,_=r.support(ev['core_start'],[actor],False)
                if support==sink:stationary.append(actor)
            if stationary:
                goal['entities']=sorted(set(goal['entities']+stationary));goal['text']='Use the faucet to rinse the '+', '.join(natural(r.es[i]) for i in stationary)+' in the sink.'
                changes.append('Include stationary food in the faucet/rinsing relevance mask.')
    if task=='ResetCabinetDoors':
        lab['goals']=[g for g in lab['goals'] if g['kind']!='open'];lab['goals'][0]['start_frame']=0
        changes.append('Suppress reverse door excursion when the manual task is close all open doors.')
    if task=='RinseCuttingBoard':
        board=[e['id'] for e in lab['entities'] if natural(e)=='cutting board'];sink=[e['id'] for e in lab['entities'] if natural(e)=='sink']
        events=event_list(lab);off=[e for e in events if e.get('joint_delta_abs',0)<0]
        assert board and sink and off
        final_off=max(off,key=lambda e:e['core_start']);opening=[e for e in events if e['core_start']<final_off['core_start']]
        assert opening
        boundary=max(e['core_end'] for e in opening)+1
        lab['goals']=[{'start_frame':0,'end_frame':boundary-1,'actors':sink,'entities':sink,'kind':'turn_on_hot_water',
                      'text':'Turn on the sink faucet and adjust it to hot water.','event':opening[0],'additional_events':opening[1:]},
                     {'start_frame':boundary,'end_frame':lab['length']-1,'actors':board,'entities':sorted(board+sink),'kind':'rinse_and_stop',
                      'text':'Rinse the cutting board under the hot water, then turn off the faucet.',
                      'evidence':'Manual stationary-rinsing subtask after water/temperature adjustment; includes demonstrated faucet stop.',
                      'closing_event':final_off}]
        changes.append('Include stationary cutting board during rinsing; split at completion of hot-water adjustment.')
    original_names={e['name'] for e in ready['entities']}
    coffee=[e for e in lab['entities'] if 'coffee machine' in natural(e) and e['name'] in original_names]
    if coffee and re.search(r'\b(press|start|brew)\b',lab['instruction'].lower()) and not '/atomic/' in ready['source_prefix']:
        r=r or Replay(D/'cases'/name);old_events=event_list(lab)
        for fixture in coffee:
            eid=fixture['id'];buttons=[g for g in r.gids[eid] if 'start_button' in (mujoco.mj_id2name(r.m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')]
            hits=[]
            for gid in buttons:
                restore_state(r.m,r.d,r.states[0]);center=r.d.geom_xpos[gid].copy()
                near=np.linalg.norm(r.eef-center,axis=1)<.085
                for t in np.flatnonzero(near)[::2]:
                    if any(e['kind']=='grasp' and e['core_start']<=t<=e['core_end'] for e in old_events):continue
                    r.contacts(int(t),[])
                    if r._direct_cache[int(t)].get(eid,0):hits.append(int(t))
            if len(set(hits))<2:continue
            begin,end=min(hits),max(hits)
            if any(e['kind']=='control' and e['primary']==eid and e['core_start']<=end+35 and e['core_end']>=begin-35 for e in old_events):continue
            event={'kind':'control','actors':[eid],'primary':eid,'core_start':begin,'core_end':end,'release':end,
                   'source':-1,'target':eid,'evidence':'Recorded gripper contact with coffee machine while within 8.5 cm of start-button geometry.',
                   'direct_hand_contact_samples':len(set(hits))}
            index=next((i for i,g in enumerate(lab['goals']) if g.get('event',{}).get('core_start',g['start_frame'])>begin),len(lab['goals']))
            start=0 if index==0 else min(begin,lab['goals'][index-1].get('event',{}).get('release',begin-1)+1)
            goal={'start_frame':start,'end_frame':lab['length']-1,'actors':[eid],'primary':eid,'entities':[eid],
                  'kind':'turn_on','text':'Press the coffee machine start button to brew coffee.','event':event}
            lab['goals'].insert(index,goal)
            if index+1<len(lab['goals']):lab['goals'][index+1]['start_frame']=end+1
            changes.append('Add visually reviewed start-button operation, grounded by recorded hand contact rather than a missing joint.')
    for a,b in zip(lab['goals'],lab['goals'][1:]):a['end_frame']=b['start_frame']-1
    assert lab['goals'] and lab['goals'][0]['start_frame']==0 and lab['goals'][-1]['end_frame']==lab['length']-1
    assert all(g['start_frame']<=g['end_frame'] and g['entities'] for g in lab['goals'])
    lab['source_annotation_version']=4;lab['version']=5;lab['review_revision']=2;lab['semantic_completion']=changes
    dest=D/'physics_v5'/name;dest.mkdir(parents=True,exist_ok=True);save(dest/'LABELS.json',lab)
    return {'name':name,'changes':changes,'goals':len(lab['goals'])}

