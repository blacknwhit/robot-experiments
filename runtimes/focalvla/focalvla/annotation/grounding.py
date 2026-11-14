"""Scene identity and motion grounding for OFFLINE annotation, not policy inputs."""
import numpy as np
import mujoco

def body_membership(model,entities):
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(model.nbody)]
    result=[]
    for e in entities:
        n=e['name'];roots=[i for i,s in enumerate(names) if s in [n,n+'_main',n+'_root']]
        if not roots:
            longer=[x['name']+'_' for x in entities if x['name']!=n and x['name'].startswith(n+'_')]
            candidates={i for i,s in enumerate(names) if s.startswith(n+'_') and not any(s.startswith(p) for p in longer)}
            roots=[i for i in candidates if int(model.body_parentid[i]) not in candidates]
        assert roots and 0 not in roots,('unresolved entity body roots',n)
        roots=set(roots);members=[]
        for i in range(1,model.nbody):
            j=i
            while j:
                if j in roots:members.append(i);break
                j=int(model.body_parentid[j])
        result.append(members)
    return result

def entity_geom_ids(model,entities):
    bodies=body_membership(model,entities)
    return [[i for i,b in enumerate(model.geom_bodyid) if int(b) in set(members)] for members in bodies]

def intervals(mask,min_length=5,merge_gap=20):
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    spans=[];start=previous=int(idx[0])
    for value in idx[1:]:
        value=int(value)
        if value-previous>merge_gap+1:
            if previous-start+1>=min_length:spans.append([start,previous])
            start=value
        previous=value
    if previous-start+1>=min_length:spans.append([start,previous])
    return spans

def motion_hints(model,states,entities):
    assert states.shape[1]==1+model.nq+model.nv+model.na
    qpos=states[:,1:1+model.nq];n=len(states);before=np.maximum(0,np.arange(n)-5);after=np.minimum(n-1,np.arange(n)+5)
    hints=[];bodies=body_membership(model,entities);gids=entity_geom_ids(model,entities)
    data=mujoco.MjData(model)
    def snapshot(frame):
        data.qpos[:]=qpos[frame];mujoco.mj_kinematics(model,data);boxes=[]
        for gs in gids:
            gs=[g for g in gs if int(model.geom_group[g])==1 and model.geom_rgba[g,3]>0] or gs
            centers=[];halves=[]
            for g in gs:
                rotation=data.geom_xmat[g].reshape(3,3)
                centers.append(data.geom_xpos[g]+rotation@model.geom_aabb[g,:3])
                halves.append(np.abs(rotation)@model.geom_aabb[g,3:])
            centers=np.asarray(centers);halves=np.asarray(halves)
            boxes.append(((centers-halves).min(0),(centers+halves).max(0)))
        return boxes
    initial_boxes=snapshot(0);final_boxes=snapshot(n-1)
    def nearby(i,boxes):
        lo,hi=boxes[i];point=(lo+hi)/2;matches=[]
        for j,(other_lo,other_hi) in enumerate(boxes):
            if j==i:continue
            distance=float(np.linalg.norm(np.maximum(np.maximum(other_lo-point,point-other_hi),0)))
            if distance<.30:matches.append((distance,float(np.prod(other_hi-other_lo)),j))
        matches.sort(key=lambda x:(round(x[0],2),x[1]))
        return [{'entity_id':j,'distance_m':round(dist,3)} for dist,_,j in matches[:3]]
    for e,members in zip(entities,bodies):
        joints=[j for j,b in enumerate(model.jnt_bodyid) if int(b) in set(members)]
        frees=[j for j in joints if int(model.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE)]
        moving=np.zeros(n,dtype=bool);large=np.zeros(n,dtype=bool);max_move=0.;max_angle=0.;first_transport=None
        if frees:
            j=frees[0];a=int(model.jnt_qposadr[j]);xyz=qpos[:,a:a+3];quat=qpos[:,a+3:a+7]
            translation=np.linalg.norm(xyz[after]-xyz[before],axis=1)
            angles=2*np.arccos(np.clip(np.abs(np.sum(quat[after]*quat[before],axis=1)),0,1))
            displacement=np.linalg.norm(xyz-xyz[0],axis=1)
            orientation=2*np.arccos(np.clip(np.abs(quat@quat[0]),0,1))
            moving=(translation>.012)|(angles>.10);large=(displacement>.04)|(orientation>.25)
            max_move=float(displacement.max());max_angle=float(orientation.max())
            supports=[j for j,other in enumerate(entities) if j!=e['id'] and (other['kind']=='fixture' or any(word in other['description'].lower() for word in ['bowl','cup','pan','pot','plate','tray','container']))]
            point=(initial_boxes[e['id']][0]+initial_boxes[e['id']][1])/2
            ranked=[]
            for j in supports:
                lo,hi=initial_boxes[j];distance=float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0)))
                if distance<.20:ranked.append((round(distance,2),float(np.prod(hi-lo)),j))
            if ranked:
                source=sorted(ranked)[0][2];lo,hi=initial_boxes[source]
                centers=xyz+(point-xyz[0]);distance=np.linalg.norm(np.maximum(np.maximum(lo-centers,centers-hi),0),axis=1)
                spans_transport=intervals((distance>distance[0]+.025)&(displacement>.04),min_length=5,merge_gap=0)
                if spans_transport:first_transport=spans_transport[0][0]
        elif e['kind']=='fixture':
            for j in joints:
                kind=int(model.jnt_type[j])
                if kind not in [int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)]:continue
                value=qpos[:,int(model.jnt_qposadr[j])]
                threshold=.04 if kind==int(mujoco.mjtJoint.mjJNT_HINGE) else .012
                moving|=np.abs(value[after]-value[before])>threshold
                large|=np.abs(value-value[0])>threshold
        spans=intervals(moving);departures=intervals(large,min_length=5,merge_gap=0)
        if spans or departures:
            hints.append({'entity_id':e['id'],'name':e['name'],'kind':e['kind'],
                'movable_free_body':bool(frees),
                'moving_intervals':spans,'first_large_change':departures[0][0] if departures else None,
                'first_transport_away_from_initial_support':first_transport,
                'max_translation_m':round(max_move,3),'max_rotation_rad':round(max_angle,3),
                'initial_nearby_entities':nearby(e['id'],initial_boxes),'final_nearby_entities':nearby(e['id'],final_boxes)})
    return {'source':'recorded simulator entity/joint motion; no subtask boundary GT',
            'purpose':'offline identity/operation consistency hints only','entities':hints}
