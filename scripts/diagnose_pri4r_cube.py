"""Bounded first-frame sampling coverage; future motion never selects points."""
import argparse,gzip,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from common import ROOT,activate


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());entries=manifest['episodes']
    if not 1<=len(entries)<=4:raise ValueError('Diagnostic bounded to four existing demonstrations')
    activate('pri4r_paper')
    import mujoco
    from starVLA.pri4r.generate import rewrite_assets,restore_state,body_pose,geom_triangles
    from starVLA.pri4r.geometry import sample_surfaces,transform_points
    from starVLA.pri4r.reader import episode_key
    cfg=json.loads((ROOT/'configs/pri4r_paper_gt.json').read_text());result=[]
    for e in entries:
        key=episode_key(e['source_prefix'],e['source_episode_index']);source=Path(e['episode_dir'])
        if not source.resolve().is_relative_to(a.asset_root.resolve()):raise ValueError('Source outside asset root')
        xml,_=rewrite_assets(gzip.decompress((source/'model.xml.gz').read_bytes()),str(a.asset_root))
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        with np.load(source/'states.npz',allow_pickle=False) as f:states=f['states']
        restore_state(model,data,states[0]);base_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg['base_body_name'])
        pose=body_pose(data,base_id);rot=data.geom_xmat.reshape(-1,3,3).copy();pos=data.geom_xpos.copy()
        selected=np.flatnonzero(np.isin(model.geom_group,cfg['geom_groups'])&(model.geom_rgba[:,3]>0))
        geometries={int(g):geom_triangles(model,int(g),primitive_segments=cfg['primitive_segments'],
            primitive_rings=cfg['primitive_rings'],plane_extent=10.) for g in selected}
        labels={};robot_parts={};robot_full={k:0. for k in ('arm_body','gripper','mobile_base')}
        for gid in selected:
            b=int(model.geom_bodyid[gid]);names=[];movable=False
            while b>0:
                names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or '')
                movable|=bool(model.body_jntnum[b]);b=int(model.body_parentid[b])
            text=' '.join(names).lower()
            robot=any(s in text for s in ('robot0','gripper0','mobilebase0'))
            labels[int(gid)]='robot' if robot else ('articulated_or_free_scene' if movable else 'static_scene')
            if robot:
                part='gripper' if 'gripper' in text else ('arm_body' if 'robot0' in text else 'mobile_base')
                robot_parts[int(gid)]=part;tri=geometries[int(gid)]
                robot_full[part]+=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1).sum()/2)
        seed=int.from_bytes(hashlib.sha256((key+':'+str(cfg['seed'])).encode()).digest()[:8],'little')
        for side in (2.,3.,4.):
            for g in selected:
                if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:
                    geometries[int(g)]=geom_triangles(model,int(g),primitive_segments=cfg['primitive_segments'],
                        primitive_rings=cfg['primitive_rings'],plane_extent=side*2+2)
            points=sample_surfaces(geometries,rot,pos,pose,count=1024,half_extent=side/2,seed=seed)
            counts={k:0 for k in ('robot','articulated_or_free_scene','static_scene')};areas=dict.fromkeys(counts,0.)
            for g in points['geom_ids']:counts[labels[int(g)]]+=1
            for g,area in points['geom_surface_area_m2'].items():areas[labels[int(g)]]+=area
            robot_kept=dict.fromkeys(robot_full,0.)
            for g,area in points['geom_surface_area_m2'].items():
                if int(g) in robot_parts:robot_kept[robot_parts[int(g)]]+=area
            identity_sha=hashlib.sha256(b''.join(points[k].tobytes() for k in ('geom_ids','face_ids','barycentric'))).hexdigest()
            # Post-sampling audit only. No rejection, resampling, quotas or seed search.
            gids=points['geom_ids'];first=None;maximum=np.zeros(1024);motion_sum=0.;pairs=0;previous=None
            for state in states:
                restore_state(model,data,state)
                world=transform_points(points['local'],data.geom_xmat[gids].reshape(-1,3,3),data.geom_xpos[gids])
                if first is None:first=world.copy()
                maximum=np.maximum(maximum,np.linalg.norm(world-first,axis=-1))
                if previous is not None:motion_sum+=float(np.abs(world-previous).sum());pairs+=world.size
                previous=world
            row={'episode':key,'cube_side_m':side,'frames':len(states),'area_m2':areas,'sampled_points':counts,
                 'surface_area_m2':points['surface_area_m2'],'moving_points_over_1mm':int((maximum>.001).sum()),
                 'robot_area_retained_fraction':{k:robot_kept[k]/v if v>0 else None for k,v in robot_full.items()},
                 'expected_point_count':{k:1024*v/points['surface_area_m2'] for k,v in areas.items()},
                 'point_identity_sha256_before_replay':identity_sha,
                 'max_displacement_m':float(maximum.max()),'zero_prediction_adjacent_xyz_mae_m':motion_sum/max(pairs,1),
                 'sampling_used_future_motion':False}
            result.append(row);print(json.dumps(row),flush=True)
        del model,data,geometries
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'rows':result,'scope':'4 existing demos; descriptive, not performance validation',
        'classification':'initial model ancestry, joints and robot naming; not task-object segmentation'},indent=2))

if __name__=='__main__':main()
