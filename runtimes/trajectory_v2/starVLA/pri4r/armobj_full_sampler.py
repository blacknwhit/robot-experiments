"""Arm + directly manipulated rigid surfaces, for offline GT only.

Episode metadata establishes object ownership; contacts establish selection.
Motion alone never admits a part. Fixed material points are sampled once.
Unknown repeated-contact ownership fails closed, rather than adding context.
"""
import re
import numpy as np
import mujoco
from .generate import restore_state
from .geometry import sample_surfaces, transform_points

SAMPLING_RULES = {
    'version': 'arm_direct_object_full_v1', 'num_points': 1024, 'seed': 20261010,
    'cube_half_extent_m': 2.0, 'contact_max_distance_m': .003,
    'minimum_contact_frames': 3, 'object_arm_background': [512, 512, 0],
    'no_contact_object_arm_background': [0, 1024, 0],
    'within_class': 'clipped_physical_surface_area', 'motion_required': False,
    'indirect_motion_selection': False, 'arm': 'links1_to_7_hand_fingers_no_base',
    'merged_housing_patch_fraction': [1, 4], 'contact_patch_half_extent_m': .05,
    'center_grid_m': .02, 'max_contact_centers': 16,
    'merged_static_housing_classes': ['Microwave', 'CoffeeMachine', 'Blender'],
    'merged_housing_requires_control_geom_contact_frames': 3,
    'structural_room_bodies_missing_from_metadata': 'exclude_wall_floor_only',
    'unknown_static_controls': 'audit_error_until_geometry_semantics_reviewed',
    'visibility_filter': False, 'selection_uses_whole_demonstration': True,
    'boundary_shortfall': 'same_rigid_object_global_crop_surface',
    'seed_policy': 'same_fixed_base_seed_per_episode',
}
SUPPORT_CLASSES = {'Wall', 'Floor', 'Counter', 'Box', 'Stool', 'Window',
                   'HousingCabinet', 'PanelCabinet', 'WallAccessory', 'Hood'}
CABINET_CLASSES = {'Drawer', 'HingeCabinet', 'SingleCabinet',
                   'Fridge', 'FridgeFrenchDoor', 'FridgeBottomFreezer',
                   'Dishwasher', 'Oven', 'Sink', 'OpenCabinet', 'FridgeSideBySide',
                   'Toaster', 'ToasterOven', 'Stove', 'Stovetop', 'ElectricKettle',
                   'StandMixer', 'Accessory'}
FIELDS = ('geom_ids', 'face_ids', 'local', 'barycentric')


def allocate(total, weights):
    keys = sorted(weights)
    if not total:
        return {}
    if not keys:
        raise ValueError('cannot allocate a positive budget to an empty pool')
    w = np.array([weights[k] for k in keys], dtype=float)
    if not np.isfinite(w).all() or not (w > 0).all():
        raise ValueError('area weights must be positive and finite')
    exact = total * w / w.sum(); n = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - n), kind='stable')[:total-int(n.sum())]:
        n[i] += 1
    return {k: int(x) for k, x in zip(keys, n) if x}


def object_owner(body_name, ep_meta):
    owners = {name: {'name': name, 'kind': 'fixture', 'class': cfg['cls']}
              for name, cfg in ep_meta['fixtures'].items()}
    for cfg in ep_meta['object_cfgs']:
        name = cfg['name']
        if name in owners:
            raise ValueError('ambiguous object and fixture name: '+name)
        owners[name] = {'name': name, 'kind': 'object', 'class': cfg.get('info', {}).get('cat', 'object')}
    matches = [name for name in owners if body_name == name or body_name.startswith(name+'_')]
    if not matches and re.fullmatch(r'(wall|floor)_[A-Za-z0-9_.]+_room_main', body_name):
        return {'name': body_name.removesuffix('_main'), 'kind': 'fixture',
                'class': 'Wall' if body_name.startswith('wall_') else 'Floor'}
    if not matches:
        raise ValueError('unresolved contacted body ownership: '+body_name)
    return owners[max(matches, key=len)]


def selection_kind(owner, has_joint):
    if owner['kind'] == 'object':
        return 'direct_object'
    cls = owner['class']
    if cls in SUPPORT_CLASSES:
        return 'excluded_support'
    if has_joint:
        return 'direct_articulated_component'
    if cls in SAMPLING_RULES['merged_static_housing_classes']:
        return 'merged_static_control_housing'
    if cls in CABINET_CLASSES:
        return 'excluded_fixed_fixture_chassis'
    raise ValueError('unreviewed directly contacted fixed fixture class: '+cls)


def sample_interaction(model, data, states, geoms, R0, p0, base, *, ep_meta):
    chosen = np.array(sorted(geoms), int)
    assert list(geoms) == chosen.tolist()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(model.nbody)]
    def ancestors(b):
        result=[]
        while b>0:
            result.append(int(b)); b=int(model.body_parentid[b])
        return result
    ancestry=[ancestors(b) for b in range(model.nbody)]
    def body_owner(b):
        # MuJoCo permits unnamed intermediate bodies. Ownership comes from the
        # nearest named scene-object ancestor, not from an empty-string prefix.
        for a in ancestry[int(b)]:
            if not names[a]:
                continue
            try:
                return object_owner(names[a],ep_meta)
            except ValueError:
                continue
        raise ValueError('unresolved scene ownership for body id '+str(b))
    robot=np.array([any(names[a].startswith(('robot0_','gripper0_','mobilebase0_')) for a in aa) for aa in ancestry])
    def rigid_part(b):
        for a in ancestry[b]:
            if model.body_jntnum[a]>0:
                return a
        return ancestry[b][-1] if ancestry[b] else 0
    part_of=np.array([rigid_part(b) for b in range(model.nbody)])
    geom_parts=part_of[model.geom_bodyid]
    hand=np.array([any(names[a]=='robot0_right_hand' or names[a].startswith('gripper0_') for a in aa) for aa in ancestry])
    corners=np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
    probes=np.stack([g.min((0,1))+(g.max((0,1))-g.min((0,1)))*corners for g in geoms.values()])
    first=np.einsum('gij,gkj->gki',R0[chosen],probes)+p0[chosen,None]
    geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(model.ngeom)]
    is_control_geom=np.array([any(k in name for k in ('start_button','stop_button','power_button')) for name in geom_names])
    control_contact_frames={}
    motion=np.zeros(model.ngeom); contact_frames={}; contact_bodies={}; contact_local={}; contact_steps=[]
    restore_state(model,data,states[0]); body_R0=data.xmat.reshape(-1,3,3).copy(); body_p0=data.xpos.copy()
    for frame,s in enumerate(states):
        restore_state(model,data,s); mujoco.mj_fwdPosition(model,data)
        moved=np.einsum('gij,gkj->gki',data.geom_xmat[chosen].reshape(-1,3,3),probes)+data.geom_xpos[chosen,None]
        motion[chosen]=np.maximum(motion[chosen],np.linalg.norm(moved-first,axis=-1).max(-1))
        contacted=set()
        for c in data.contact:
            if c.dist>SAMPLING_RULES['contact_max_distance_m']:
                continue
            b1,b2=map(int,model.geom_bodyid[np.asarray(c.geom,dtype=int)])
            if bool(hand[b1])==bool(hand[b2]):
                continue
            b=b2 if hand[b1] else b1
            if robot[b]:
                continue
            part=int(part_of[b]); contacted.add(part)
            other_g=int(c.geom[1] if hand[b1] else c.geom[0])
            if is_control_geom[other_g]:
                control_contact_frames.setdefault(part,set()).add(frame)
            contact_bodies.setdefault(part,{}).setdefault(b,set()).add(frame)
            contact_local.setdefault(part,[]).append((np.asarray(c.pos)-data.xpos[part])@data.xmat[part].reshape(3,3))
        for part in contacted:
            contact_frames.setdefault(part,set()).add(frame)
        contact_steps.append(sorted(contacted))

    groups={}
    for g in chosen:
        if not robot[model.geom_bodyid[g]]:
            groups.setdefault(int(geom_parts[g]),[]).append(int(g))
    eligible={}; owners={}; kinds={}; areas={}; exclusions=[]
    def sample(gs,count,seed):
        return sample_surfaces({g:geoms[g] for g in gs},R0,p0,base,count=count,half_extent=2,seed=seed)
    for b,frames in sorted(contact_frames.items()):
        if len(frames)<SAMPLING_RULES['minimum_contact_frames']:
            exclusions.append({'part':names[b],'reason':'fewer_than_3_contact_frames','contact_frames':len(frames)})
            continue
        owner=body_owner(b)
        # A rigid group is not allowed to cross scene-object boundaries.
        for body in set(map(int,model.geom_bodyid[groups.get(b,[])])) | set(contact_bodies[b]):
            if body_owner(body)!=owner:
                raise ValueError('rigid group crosses scene-object ownership: '+names[b])
        kind=selection_kind(owner,bool(model.body_jntnum[b]))
        if kind.startswith('excluded_'):
            exclusions.append({'part':names[b],'owner':owner,'reason':kind,'contact_frames':len(frames)})
            continue
        if kind=='merged_static_control_housing' and len(control_contact_frames.get(b,()))<3:
            exclusions.append({'part':names[b],'owner':owner,'reason':'no_repeated_control_button_contact',
                               'contact_frames':len(frames),'button_contact_frames':len(control_contact_frames.get(b,()))})
            continue
        gs=groups.get(b,[])
        if not gs:
            raise ValueError('contacted target has no eligible visual surface: '+names[b])
        try:
            area=sample(gs,1,3)['surface_area_m2']
        except ValueError as exc:
            exclusions.append({'part':names[b],'reason':'outside_initial_crop','details':str(exc)})
            continue
        eligible[b]=gs; owners[b]=owner; kinds[b]=kind; areas[b]=area
    object_counts=allocate(512,areas) if eligible else {}
    samples=[]; categories=[]; allocations=[]; patch_reports=[]; seed=SAMPLING_RULES['seed']
    def add(gs,count,label,kind,part=None):
        nonlocal seed
        if count<=0:
            return
        seed+=1; s=sample(gs,count,seed); samples.append({k:s[k] for k in FIELDS}); categories.extend([label]*count)
        allocations.append({'kind':kind,'part':names[part] if part is not None else None,'count':count,
                            'surface_area_m2':s['surface_area_m2'],'sampled_geoms':s['selected_geom_count']})
    for b,count in sorted(object_counts.items()):
        gs=eligible[b]
        if kinds[b]!='merged_static_control_housing':
            add(gs,count,2,'direct_object_area',b)
            continue
        focused=count//4; centers=np.asarray(contact_local[b]); actual=0; patches=[]
        _,unique=np.unique(np.rint(centers/.02).astype(np.int64),axis=0,return_index=True)
        centers=centers[np.sort(unique)]
        if len(centers)>16:
            centers=centers[np.linspace(0,len(centers)-1,16,dtype=int)]
        for center in centers:
            frame=np.eye(4); frame[:3,:3]=body_R0[b]; frame[:3,3]=body_R0[b]@center+body_p0[b]; seed+=1
            try:
                pool=sample_surfaces({g:geoms[g] for g in gs},R0,p0,frame,count=max(512,focused*2),half_extent=.05,seed=seed)
            except ValueError:
                continue
            world=transform_points(pool['local'],R0[pool['geom_ids']],p0[pool['geom_ids']])
            keep=np.all(np.abs((world-base[:3,3])@base[:3,:3])<=2+1e-9,axis=-1)
            if keep.any():
                patches.append({k:pool[k][keep] for k in FIELDS})
        if patches and focused:
            rng=np.random.default_rng(seed+1)
            for i,n in allocate(focused,{i:1 for i in range(len(patches))}).items():
                pool=patches[i]; n=min(n,len(pool['local'])); ids=rng.choice(len(pool['local']),size=n,replace=False)
                samples.append({k:pool[k][ids] for k in FIELDS}); categories.extend([2]*n); actual+=n
            allocations.append({'kind':'merged_housing_contact_patch','part':names[b],'count':actual})
        add(gs,count-actual,2,'merged_housing_area_remainder',b)
        patch_reports.append({'part':names[b],'requested_focused_points':focused,'focused_points':actual,
                              'shortfall_same_object_fill':focused-actual,'valid_patches':len(patches),
                              'half_extent_m':.05,'contact_centers_material_frame':centers.tolist()})

    allowed_links={'robot0_link'+str(i) for i in range(1,8)}
    arm=[int(g) for g in chosen if names[int(model.geom_bodyid[g])] in allowed_links or hand[int(model.geom_bodyid[g])]]
    assert arm, 'expected arm geometry missing'
    add(arm,512 if eligible else 1024,1,'arm_hand_area')
    new={k:np.concatenate([s[k] for s in samples]) for k in FIELDS}; labels=np.asarray(categories,np.int8); gids=new['geom_ids']
    assert len(gids)==1024 and not (labels==0).any()
    assert (labels==2).sum()==(512 if eligible else 0)
    assert len(np.unique(np.concatenate([gids[:,None],new['local']],axis=1),axis=0))==1024
    assert np.all(new['barycentric']>=-1e-12) and np.allclose(new['barycentric'].sum(-1),1)
    for g in np.unique(gids):
        mask=gids==g
        assert np.allclose(np.einsum('ni,nij->nj',new['barycentric'][mask],geoms[int(g)][new['face_ids'][mask]]),new['local'][mask],atol=1e-12)
    world=transform_points(new['local'],R0[gids],p0[gids]); assert np.abs((world-base[:3,3])@base[:3,:3]).max()<=2+1e-9
    info={'rule_version':SAMPLING_RULES['version'],'object_primary_quota':{names[b]:n for b,n in object_counts.items()},
          'object_related_quota':{},'allocations':allocations,'contact_patch_reports':patch_reports,
          'selection_mode':'repeated_direct_hand_contact' if eligible else 'no_eligible_direct_object',
          'fallback':None if eligible else '1024 arm points; zero object/background points',
          'visibility_filter_in_gt':False,'contact_uses_future_demonstration':True,'contact_or_future_labels_are_policy_input':False,
          'parts':[{'part':names[b],'contact_frames':len(contact_frames.get(b,())),
                    'motion_probe_m':float(motion[gs].max()),'selected':b in eligible,
                    'control_button_contact_frames':len(control_contact_frames.get(b,())),
                    'owner':owners.get(b),'selection_kind':kinds.get(b),
                    'surface_area_m2':areas.get(b),'contact_bodies':{names[x]:len(fr) for x,fr in contact_bodies.get(b,{}).items()}}
                   for b,gs in sorted(groups.items()) if len(contact_frames.get(b,())) or motion[gs].max()>.002],
          'excluded':exclusions}
    object_geoms={g for gs in eligible.values() for g in gs}
    geom_categories=np.array([1 if g in arm else 2 if g in object_geoms else 0 for g in range(model.ngeom)])
    return new,labels,geom_categories,info,contact_steps
