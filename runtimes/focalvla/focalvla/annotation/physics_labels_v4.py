"""Offline coarse subgoals grounded by grasp rigidity, release and contact.

The policy never receives these simulator variables. This is an annotation
enhancement; human semantic templates and actual demonstrated entities are kept.
"""
from pathlib import Path
import sys,json,re,collections,math,time
from .prepare import D
from .prepare import make_sim,SOURCE,restore_state,mujoco
from .grounding import body_membership,entity_geom_ids,intervals
import numpy as np

STRUCTURAL={'Wall','Floor','WallAccessory','Accessory','Box','HousingCabinet','Hood','Stool','Chair'}
TOOLS=['spatula','spoon','ladle','whisk','fork','tong','sponge','cloth','brush','knife','cutter','peeler','mallet','rolling_pin']
CONTAINERS=['bowl','cup','mug','pot','pan','plate','tray','container','jug','kettle','bottle','pitcher','board','basket','colander','tupperware','jar','strainer','teapot','saucepan','glass']
SUPPORTS=CONTAINERS+['scale','reamer','grater','rack']

def contains_word(text,words):
    return any(re.search(r'\b'+re.escape(w)+r'(?:s|es)?\b',text) for w in words)

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def multiply_quats(a,b):
    aw,ax,ay,az=a.T;bw,bx,by,bz=b.T
    return np.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],axis=1)

def natural(e):
    x=e['description']
    for name in ['HingeCabinet','SingleCabinet','OpenCabinet','PanelCabinet']:x=x.replace(name,'cabinet')
    x=x.replace('canned_food','can')
    x=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',x).replace('_',' ').lower()
    if x=='counter' and 'island' in e['name']:x='island counter'
    return x

class Replay:
    def __init__(self,case):
        self.case=case;self.info=json.loads((case/'READY.json').read_text());i=self.info
        path=SOURCE/i['source_prefix']/'lerobot/extras'/f"episode_{i['source_episode_index']:06d}"
        self.m,self.d,self.states,self.meta=make_sim(path);self.n=len(self.states)
        self.es=[dict(e) for e in i['entities']];names={e['name'] for e in self.es};self.unresolved=[]
        for name,cfg in self.meta.get('fixtures',{}).items():
            cls=cfg.get('cls','')
            if name in names or cls in STRUCTURAL or 'Housing' in cls:continue
            e={'id':len(self.es),'name':name,'description':cls,'kind':'fixture','aliases':[]}
            try:body_membership(self.m,[e])
            except AssertionError:self.unresolved.append(name);continue
            self.es.append(e);names.add(name)
        # A task can explicitly require rearranging existing scene objects.
        # Their old distractor name does not make an actually moved object irrelevant.
        for cfg in self.meta.get('object_cfgs',[]):
            name=cfg['name']
            if name in names:continue
            e={'id':len(self.es),'name':name,'description':cfg.get('info',{}).get('cat',str(cfg.get('obj_groups',name))),
               'kind':'object','aliases':['scene object; used only when interaction evidence selects it']}
            try:body_membership(self.m,[e])
            except AssertionError:self.unresolved.append(name);continue
            self.es.append(e);names.add(name)
        self.members=body_membership(self.m,self.es);self.gids=entity_geom_ids(self.m,self.es)
        self.owner=np.full(self.m.ngeom,-1,dtype=np.int32)
        for i in sorted(range(len(self.es)),key=lambda i:-len(self.gids[i])):self.owner[self.gids[i]]=i
        self.qpos=self.states[:,1:1+self.m.nq];self.frees={};self.joints={}
        for e,bs in zip(self.es,self.members):
            js=[j for j,b in enumerate(self.m.jnt_bodyid) if int(b) in set(bs)];self.joints[e['id']]=js
            free=[j for j in js if int(self.m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE)]
            if free:self.frees[e['id']]=int(self.m.jnt_qposadr[free[0]])
        fingers=[j for j in range(self.m.njnt) if 'finger_joint' in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,j) or '')]
        assert len(fingers)==2
        self.opening=sum(np.abs(self.qpos[:,int(self.m.jnt_qposadr[j])]) for j in fingers)
        maximum=sum(float(np.abs(self.m.jnt_range[j]).max()) for j in fingers)
        self.is_open=self.opening>.95*maximum
        self.closed=intervals(~self.is_open,min_length=4,merge_gap=2)
        sites=[j for j in range(self.m.nsite) if (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_SITE,j) or '').endswith('grip_site')]
        assert len(sites)==1;site=sites[0]
        self.eef=np.empty((self.n,3));self.eef_rot=np.empty((self.n,3,3));self.eef_quat=np.empty((self.n,4))
        for t,row in enumerate(self.states):
            restore_state(self.m,self.d,row)
            self.eef[t]=self.d.site_xpos[site];self.eef_rot[t]=self.d.site_xmat[site].reshape(3,3)
            mujoco.mju_mat2Quat(self.eef_quat[t],self.d.site_xmat[site])
        self._bounds_cache={};self._contacts_cache={};self._direct_cache={}
        self.gripper_geoms=np.array(['gripper' in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')
            or 'gripper' in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,int(self.m.geom_bodyid[g])) or '') for g in range(self.m.ngeom)])

    def bounds(self,t,i):
        key=(int(t),i)
        if key in self._bounds_cache:return self._bounds_cache[key]
        restore_state(self.m,self.d,self.states[t]);gs=[g for g in self.gids[i] if self.m.geom_group[g]==1 and self.m.geom_rgba[g,3]>0] or self.gids[i]
        cs=[];hs=[]
        for g in gs:
            r=self.d.geom_xmat[g].reshape(3,3);cs.append(self.d.geom_xpos[g]+r@self.m.geom_aabb[g,:3]);hs.append(np.abs(r)@self.m.geom_aabb[g,3:])
        cs=np.asarray(cs);hs=np.asarray(hs);value=((cs-hs).min(0),(cs+hs).max(0));self._bounds_cache[key]=value;return value

    def contacts(self,t,actors):
        t=int(np.clip(t,0,self.n-1))
        if t not in self._contacts_cache:
            restore_state(self.m,self.d,self.states[t]);mujoco.mj_forward(self.m,self.d)
            self._contacts_cache[t]=[(int(self.owner[c.geom1]),int(self.owner[c.geom2]),float(c.frame[2]))
                for c in self.d.contact if c.dist<.008]
            direct=collections.Counter()
            for c in self.d.contact:
                if c.dist>=.008:continue
                a,b=int(self.owner[c.geom1]),int(self.owner[c.geom2])
                if self.gripper_geoms[c.geom1] and b>=0:direct[b]+=1
                if self.gripper_geoms[c.geom2] and a>=0:direct[a]+=1
            self._direct_cache[t]=direct
        count=collections.Counter();vertical=collections.Counter()
        for a,b,z in self._contacts_cache[t]:
            if a in actors and b>=0 and b not in actors:count[b]+=1;vertical[b]+=int(z<-.45)
            if b in actors and a>=0 and a not in actors:count[a]+=1;vertical[a]+=int(z>.45)
        return count,vertical

    def support(self,t,actors,promote=True):
        count=collections.Counter();vertical=collections.Counter()
        for k in [0,3,8]:
            c,v=self.contacts(min(self.n-1,t+k),actors);count.update(c);vertical.update(v)
        if count:
            ranked=sorted(count,key=lambda j:(vertical[j]>0,self.es[j]['kind']=='object',vertical[j],count[j]),reverse=True)
            chosen=ranked[0]
            # A cube can physically rest on a previous cube inside the blender.
            # For a non-stacking goal the relevant recipient is the enclosing
            # container, not that incidental support object.
            desc=natural(self.es[chosen]);text=self.info['instruction'].lower()
            explicit_target=bool(re.search(r'\b(on|onto|in|into|inside)\b[^.!?]{0,35}\b'+re.escape(desc)+r'\b',text))
            if promote and self.es[chosen]['kind']=='object' and not contains_word(desc,SUPPORTS) and not re.search(r'\bstack\b',text) and not explicit_target:
                parent,chain,_=self.support(t,actors+[chosen],False)
                if parent>=0 and (self.es[parent]['kind']=='fixture' or contains_word(natural(self.es[parent]),SUPPORTS)):
                    return parent,sorted(set(ranked[:4]+chain)), 'contact_hierarchy'
            return chosen,ranked[:4],'contact'
        # Conservative geometric fallback for an already held/just released item.
        lo,hi=self.bounds(t,actors[0]);point=(lo+hi)/2;scores=[]
        for e in self.es:
            j=e['id']
            if j in actors:continue
            a,b=self.bounds(t,j);desc=natural(e)
            xy=float(np.linalg.norm(np.maximum(np.maximum(a[:2]-point[:2],point[:2]-b[:2]),0)))
            if any(w in desc for w in ['counter','plate','tray','stool']):distance=xy+abs(float(lo[2]-b[2]))
            else:distance=float(np.linalg.norm(np.maximum(np.maximum(a-point,point-b),0)))
            if distance<.12:scores.append((round(distance,2),float(np.prod(b-a)),j))
        if scores:
            scores.sort();return scores[0][2],[x[2] for x in scores[:3]],'geometry'
        return -1,[],'unresolved'

    def grasp_events(self):
        a=np.maximum(0,np.arange(self.n)-3);b=np.minimum(self.n-1,np.arange(self.n)+3)
        qc=self.eef_quat.copy();qc[:,1:]*=-1;activity={}
        for i,offset in self.frees.items():
            xyz=self.qpos[:,offset:offset+3];quat=self.qpos[:,offset+3:offset+7]
            rel=np.einsum('tji,tj->ti',self.eef_rot,xyz-self.eef);qr=multiply_quats(qc,quat)
            relative_motion=np.linalg.norm(rel[b]-rel[a],axis=1)
            relative_angle=2*np.arccos(np.clip(np.abs(np.sum(qr[b]*qr[a],axis=1)),0,1))
            movement=np.linalg.norm(xyz[b]-xyz[a],axis=1)
            rotation=2*np.arccos(np.clip(np.abs(np.sum(quat[b]*quat[a],axis=1)),0,1))
            lo,hi=self.bounds(0,i);radius=float(np.linalg.norm(hi-lo)/2)+.16
            activity[i]=((relative_motion<.025)&(relative_angle<.18)&((movement>.004)|(rotation>.04))&
                         (np.linalg.norm(xyz-self.eef,axis=1)<radius))
        # Wide objects may be held with the fingers almost fully open. Actual
        # hand-object contact therefore extends the kinematic close intervals.
        free_touch=np.zeros(self.n,dtype=bool)
        for t in range(0,self.n,5):
            self.contacts(t,[])
            if any(i in self.frees for i in self._direct_cache[t]):free_touch[t:min(self.n,t+5)]=True
        any_rigid_motion=np.any(np.stack(list(activity.values())),axis=0) if activity else np.zeros(self.n,dtype=bool)
        # Release can occur without reaching the maximum finger opening. The
        # absence of both hand contact and rigid object motion separates grasps.
        episodes=intervals(free_touch|((~self.is_open)&any_rigid_motion),min_length=4,merge_gap=8)
        events=[]
        for start,end in episodes:
            counts={i:int(v[start:end+1].sum()) for i,v in activity.items()}
            direct=collections.Counter();direct_times=collections.defaultdict(list)
            for t in range(start,end+1,5):
                self.contacts(t,[])
                for i in self._direct_cache[t]:
                    if i in self.frees:direct[i]+=1;direct_times[i].append(t)
            moving=[]
            for i in direct:
                offset=self.frees[i];xyz=self.qpos[start:end+1,offset:offset+3];q=self.qpos[start:end+1,offset+3:offset+7]
                displacement=float(np.linalg.norm(xyz-xyz[0],axis=1).max())
                angle=float((2*np.arccos(np.clip(np.abs(q@q[0]),0,1))).max())
                if direct[i]>=2 and (displacement>.01 or angle>.08):moving.append(i)
            if moving:
                primary=max(moving,key=lambda i:(direct[i],counts[i]));active=np.asarray(direct_times[primary],dtype=int)
            elif counts and max(counts.values())>=20:
                primary=max(counts,key=counts.get);active=np.flatnonzero(activity[primary][start:end+1])+start
            else:continue
            top=max(8,counts[primary]);group=[primary]
            for i,count in counts.items():
                if i==primary or count<max(8,.70*top):continue
                overlap=int((activity[i][start:end+1]&activity[primary][start:end+1]).sum())
                if overlap/min(count,top)>.7:group.append(i)
            if len(group)>1:
                touched=[i for i in group if direct[i]>=max(2,.5*direct[primary])]
                if touched:primary=max(touched,key=lambda i:(direct[i],counts[i]))
            group=sorted(group)
            source,source_contacts,source_evidence=self.support(max(0,int(active[0])-10),group)
            target,target_contacts,target_evidence=self.support(min(self.n-1,end+8),group)
            interactions=collections.Counter()
            for t in np.linspace(int(active[0]),int(active[-1]),7,dtype=int):
                c,_=self.contacts(int(t),group);interactions.update(c)
            related=[i for i,_ in interactions.most_common(5)]
            event={'kind':'grasp','actors':group,'primary':primary,'core_start':int(active[0]),'core_end':int(active[-1]),
                'closed_start':start,'release':min(self.n-1,end+1),'source':source,'target':target,
                'source_evidence':source_evidence,'target_evidence':target_evidence,
                'contact_entities':sorted(set(source_contacts+target_contacts+related)),
                'rigid_motion_frames':counts[primary],'direct_gripper_contact_samples':direct[primary],
                'all_candidate_scores':counts}
            # Container contents need not remain rigid relative to the hand:
            # a tilted bowl releases food while the bowl is still grasped.
            if contains_word(natural(self.es[primary]),CONTAINERS):
                t0=max(0,int(active[0])-10);t1=min(self.n-1,end+8)
                initial,_=self.contacts(t0,[primary]);contents=[];transfers=[]
                for other in initial:
                    if other not in self.frees or other in group:continue
                    below,_,_=self.support(t0,[other],False)
                    if below!=primary:continue
                    contents.append(other)
                    dest,_,_=self.support(t1,[other])
                    off=self.frees[other]
                    moved=float(np.linalg.norm(self.qpos[t1,off:off+3]-self.qpos[t0,off:off+3]))>.04
                    if dest>=0 and dest!=primary and moved:transfers.append({'entity':other,'target':dest})
                event['initial_contents']=contents;event['transferred_contents']=transfers
                event['contact_entities']=sorted(set(event['contact_entities']+contents))
            events.append(event)
        return events

    def control_events(self,grasps):
        result=[];a=np.maximum(0,np.arange(self.n)-3);b=np.minimum(self.n-1,np.arange(self.n)+3)
        for e in self.es:
            i=e['id']
            if e['kind']!='fixture' or i in self.frees:continue
            for j in self.joints[i]:
                kind=int(self.m.jnt_type[j])
                if kind not in [int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)]:continue
                q=self.qpos[:,int(self.m.jnt_qposadr[j])];span=float(np.ptp(self.m.jnt_range[j]))
                threshold=max(.005,min(.04,span*.025)) if kind==int(mujoco.mjtJoint.mjJNT_HINGE) else max(.0002,min(.008,span*.025))
                for start,end in intervals(np.abs(q[b]-q[a])>threshold,min_length=3,merge_gap=12):
                    middle=(start+end)//2
                    if any(g['core_start']<=middle<=g['core_end'] and g['direct_gripper_contact_samples']>=2 for g in grasps):continue
                    lo,hi=self.bounds(middle,i)
                    if np.linalg.norm(np.maximum(np.maximum(lo-self.eef[middle],self.eef[middle]-hi),0))>.30:continue
                    joint=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
                    delta=float(abs(q[min(self.n-1,end+3)])-abs(q[max(0,start-3)]))
                    articulated=any(word in joint.lower() for word in ['door','drawer','rack','lid','head']) or contains_word(natural(e),['cabinet','drawer','fridge'])
                    # Small contact-induced reversals do not define a new goal.
                    if articulated and abs(delta)<(max(.20,span*.10) if kind==int(mujoco.mjtJoint.mjJNT_HINGE) else max(.06,span*.12)):continue
                    touches=0
                    for t in np.linspace(start,end,min(7,end-start+1),dtype=int):
                        self.contacts(int(t),[]);touches+=int(self._direct_cache[int(t)].get(i,0)>0)
                    # Ignore scene initialization settling unrelated to the hand.
                    if start<3 and touches==0:continue
                    result.append({'kind':'control','actors':[i],'primary':i,'core_start':start,'core_end':end,'release':end,
                        'joint_name':joint,'dominant_joint':joint,'joint_delta_abs':delta,'normalized_amplitude':abs(delta)/max(span,.001),
                        'source':-1,'target':i,'contact_entities':[],'direct_hand_contact_samples':touches})
        # One operation often moves several coupled joints together.
        merged=[]
        for event in sorted(result,key=lambda e:e['core_start']):
            if merged and event['primary']==merged[-1]['primary'] and event['core_start']<=merged[-1]['core_end']+15:
                merged[-1]['core_end']=max(merged[-1]['core_end'],event['core_end']);merged[-1]['release']=merged[-1]['core_end']
                merged[-1]['joint_name']+=' | '+event['joint_name'];merged[-1]['joint_delta_abs']+=event['joint_delta_abs']
                if event['normalized_amplitude']>merged[-1]['normalized_amplitude']:
                    merged[-1]['dominant_joint']=event['dominant_joint'];merged[-1]['normalized_amplitude']=event['normalized_amplitude']
            else:merged.append(event)
        return merged

def semantic(event,r):
    e=r.es[event['primary']];actor=natural(e);src=event['source'];dst=event['target'];text=r.info['instruction'].lower();task=r.info['task']
    noun=lambda i:natural(r.es[i]) if i>=0 else 'the appropriate location'
    related=set(event['actors']+event.get('initial_contents',[])+([src] if src>=0 else [])+([dst] if dst>=0 else []))
    kind='transfer'
    if event['kind']=='control':
        joint=event.get('dominant_joint',event['joint_name']).lower();desc=natural(e)
        if 'rack' in joint:kind='slide';phrase=f'Reposition the rack of the {actor}.'
        elif any(w in joint for w in ['door','drawer','lid','head']) or any(w in desc for w in ['cabinet','fridge','drawer']):
            kind='open' if event['joint_delta_abs']>0 else 'close';phrase=f'{kind.capitalize()} the {actor}.'
        elif task.startswith('TurnOn') or task.startswith('Start'):kind='turn_on';phrase=f'Turn on the {actor}.'
        elif task.startswith('TurnOff'):kind='turn_off';phrase=f'Turn off the {actor}.'
        elif 'temperature' in text or 'spout' in text or task.startswith('Adjust') or task=='LowerHeat':kind='adjust';phrase=f'Adjust the {actor} as instructed.'
        elif 'sink' in desc:kind='operate';phrase=f'Operate the faucet of the {actor} for the current task.'
        else:kind='operate';phrase=f'Operate the {actor} as instructed.'
        contacts,_=r.contacts(event['core_start'],event['actors'])
        for j in contacts:
            if 'lid' in natural(r.es[j]) and any(word in natural(r.es[j]) for word in actor.split()):related.add(j)
    else:
        is_tool=contains_word(actor,TOOLS);is_container=contains_word(actor,CONTAINERS)
        semantic_targets=[]
        def matching(words):return [x['id'] for x in r.es if x['id'] not in event['actors'] and any(w in natural(x) for w in words)]
        if event.get('transferred_contents'):
            transfers=event['transferred_contents'];food=', '.join(noun(x['entity']) for x in transfers)
            destinations=', '.join(sorted(set(noun(x['target']) for x in transfers)))
            kind='transfer_contents';phrase=f'Transfer the {food} from the {actor} to the {destinations}.'
            semantic_targets=[x[k] for x in transfers for k in ['entity','target']]
        elif 'lid' in actor:
            kind='lid';phrase=f'Move the {actor} from the {noun(src)} to the {noun(dst)}.'
        elif is_tool and re.search(r'\b(stir|mix)\b',text):
            kind='stir';semantic_targets=matching(['pot','bowl','pan']);phrase=f'Use the {actor} to stir the ingredients as instructed.'
        elif is_tool and re.search(r'\bpress\b',text):
            kind='press';semantic_targets=[x['id'] for x in r.es if x['kind']=='object' and x['id'] not in event['actors'] and any(w in natural(x) for w in ['chicken','meat','fish'])];phrase=f'Use the {actor} to press the instructed food.'
        elif re.search(r'\b(juice|squeeze)\b',text) and matching(['reamer']):
            kind='juice';semantic_targets=matching(['reamer']);phrase=f'Juice the {actor} using the reamer.'
        elif is_tool and re.search(r'\b(scrub|clean|wipe)\b',text):
            kind='scrub';semantic_targets=matching(['microwave'] if 'microwave' in text else ['bowl'] if 'bowl' in text else ['board'] if 'board' in text else ['sink']);phrase=f'Use the {actor} to clean the instructed object or surface.'
        elif is_container and re.search(r'\bfill\b',text) and 'water' in text:
            kind='fill';semantic_targets=matching(['sink']);phrase=f'Fill the {actor} with water at the sink.'
        elif re.search(r'\b(rinse|wash)\b',text) and not is_tool:
            kind='rinse';semantic_targets=matching(['sink']);phrase=f'Rinse the {actor} at the sink.'
        elif is_container and (re.search(r'\b(dump|pour)\b',text) or re.search(r'(?:^|[.;]|\band\b|\bthen\b)\s*empty\b',text)):
            kind='pour';semantic_targets=[i for i in event['contact_entities'] if i not in event['actors']];phrase=f'Empty the contents of the {actor} into the instructed destination.'
        elif is_container and re.search(r'\b(tilt|rotate|shake)\b',text):kind='tilt';phrase=f'Tilt, rotate or shake the {actor} as instructed.'
        elif dst>=0 and 'scale' in noun(dst):kind='weigh';phrase=f'Place the {actor} on the scale to weigh it.'
        elif re.search(r'\bstack\b',text) and dst>=0 and r.es[dst]['kind']=='object':kind='stack';phrase=f'Stack the {actor} on the {noun(dst)}.'
        else:
            subjects=('the '+actor+' and its carried objects') if len(event['actors'])>1 else 'the '+actor
            phrase=f'Move {subjects}'+(f' from the {noun(src)}' if src>=0 else '')+(f' to the {noun(dst)}' if dst>=0 else ' as instructed')+'.'
        # Only explicitly task-related matches are used; room-wide same-class
        # fixtures are not added as a semantic target without geometric evidence.
        close=set(event['contact_entities'])|{src,dst}
        semantic_targets=[i for i in semantic_targets if r.es[i]['kind']=='object' or i in close or any(alias for alias in r.es[i].get('aliases',[]))]
        related.update(semantic_targets)
        # Relational goals (e.g. a mug beside a breakfast plate) need the named
        # reference object even when the physical support is a counter.
        clauses=re.findall(r'\b(?:beside|next to|near|between)\b([^.;]+)',text)
        references=collections.defaultdict(list)
        for other in r.es:
            if other['id'] in event['actors'] or other['kind']!='object':continue
            word=natural(other).split()[-1]
            if any(re.search(r'\b'+re.escape(word)+r's?\b',clause) for clause in clauses):references[natural(other)].append(other['id'])
        at=min(r.n-1,event['release']+8);lo,hi=r.bounds(at,event['primary']);center=(lo+hi)/2
        for candidates in references.values():
            if len(candidates)==1 or 'between' in text:related.update(candidates);continue
            def distance(i):
                a,b=r.bounds(at,i);return np.linalg.norm(np.maximum(np.maximum(a-center,center-b),0))
            related.add(min(candidates,key=distance))
    return kind,phrase,sorted(i for i in related if i>=0)

def process(case):
    start=time.time();r=Replay(case);grasps=r.grasp_events();controls=r.control_events(grasps)
    # Moving drawer contents are not necessarily grasped by the hand.
    grasps=[g for g in grasps if g['direct_gripper_contact_samples']>=2 or not any(
        max(g['core_start'],c['core_start'])<=min(g['core_end'],c['core_end']) for c in controls)]
    # A failed pickup can briefly move the source bowl together with a cube.
    # If only the ingredient is requested and its real transfer follows, keep
    # that retry inside the ingredient goal rather than inventing a bowl goal.
    text=r.info['instruction'].lower()
    spans=re.findall(r'\b(?:put|place|move|carry|bring|transfer|arrange|stack|sort|collect|gather|remove|retrieve|rinse|wash|scrub|press|tilt|rotate|shake|dump|empty|fill|return|pick\s+up|take)\s+([^.;]+)',text)
    spans=[re.split(r'\b(?:from|into|inside|onto|on|in|to|out of)\b',s)[0] for s in spans]
    expected={i for i in r.frees if any(re.search(r'\b'+re.escape(natural(r.es[i]).split()[-1])+r'(?:s|es)?\b',s) for s in spans)}
    filtered=[]
    for index,g in enumerate(grasps):
        if expected and g['primary'] not in expected:
            upcoming=next((h for h in grasps[index+1:] if h['primary'] in expected and
                (h['primary'] in g['actors'] or h['source']==g['primary'])),None)
            if upcoming is not None and g['source']==g['target']:
                upcoming['preparatory_retry']=g;continue
        filtered.append(g)
    grasps=filtered
    # A brief same-support retry belongs to the next grasp of the same object.
    filtered=[]
    for index,g in enumerate(grasps):
        later=next((h for h in grasps[index+1:] if h['primary']==g['primary']),None)
        if g['source']==g['target'] and g['rigid_motion_frames']<15 and later is not None:
            later['preparatory_retry']=g;continue
        filtered.append(g)
    grasps=filtered
    events=sorted(grasps+controls,key=lambda e:(e['core_start'],e['core_end']))
    if r.info['task']=='NavigateKitchen':
        target=next(e['id'] for e in r.es if 'target_fixture' in e.get('aliases',[]))
        goals=[{'start_frame':0,'end_frame':r.n-1,'actors':[],'entities':[target],
            'kind':'navigate','text':f'Navigate to the {natural(r.es[target])}.','evidence':'recorded target fixture'}]
    else:
        goals=[];previous_end=-1
        for event in events:
            kind,text,entities=semantic(event,r)
            if goals and goals[-1]['kind']==kind and goals[-1]['text']==text and goals[-1]['entities']==entities:
                goals[-1].setdefault('additional_events',[]).append(event);previous_end=max(previous_end,event['release']);continue
            begin=0 if not goals else min(event['core_start'],previous_end+1)
            begin=max(begin,goals[-1]['start_frame']+1 if goals else 0)
            if begin>=r.n:continue
            goals.append({'start_frame':begin,'end_frame':r.n-1,'actors':event['actors'],'primary':event['primary'],
                'entities':entities,'kind':kind,'text':text,'event':event})
            previous_end=max(previous_end,event['release'])
        for a,b in zip(goals,goals[1:]):a['end_frame']=b['start_frame']-1
        if '/atomic/' in r.info['source_prefix']:
            # An atomic task is itself a valid manually specified coarse goal;
            # this also covers button activation without a moving joint.
            ids=sorted(set(i for g in goals for i in g['entities'])) or [e['id'] for e in r.info['entities']]
            goals=[{'start_frame':0,'end_frame':r.n-1,'actors':[],'entities':ids,'kind':'atomic_goal',
                'text':r.info['instruction'],'evidence':'single atomic task instruction with its declared entities'}]
        if r.info['task']=='RotatePan':
            actors=[e['id'] for e in r.info['entities'] if contains_word(natural(e),['pan'])]
            ids=set(actors)
            for actor in actors:
                c,_=r.contacts(r.n//2,[actor]);ids.update(c)
            goals=[{'start_frame':0,'end_frame':r.n-1,'actors':actors,'entities':sorted(ids),'kind':'rotate',
                    'text':r.info['instruction'],'evidence':'manually specified single rotation subtask; pushing does not require a closed grasp'}]
    out={'name':r.info['name'],'task':r.info['task'],'instruction':r.info['instruction'],'length':r.n,
        'entities':r.es,'goals':goals,'grasp_count':len(grasps),'control_count':len(controls),
        'status':'candidate' if goals else 'needs_review_no_physical_event','seconds':time.time()-start,
        'annotation_method':'coarse manual semantic templates aligned with replayed grasp rigidity, release and contact',
        'simulator_inputs_offline_only':True,'manual_frame_boundaries_used':False,'unresolved_unused_fixture_roots':r.unresolved,
        'source_ready_version':r.info.get('version'),'human_review':'pending'}
    out['version']=4
    dest=D/'physics_v4'/r.info['name'];dest.mkdir(parents=True,exist_ok=True);save(dest/'LABELS.json',out)
    print('PHYSICS',r.info['name'],'grasp',len(grasps),'control',len(controls),'goals',len(goals),'seconds',round(time.time()-start,2),flush=True)
    return out

