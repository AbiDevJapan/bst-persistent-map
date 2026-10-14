"""Step 9.7B: deterministic robust 3-DoF scan-to-native-surfel alignment.

Only x,y,yaw vary. The incoming SE(3) pose supplies fixed roll,pitch,z.
Map centroids, second moments (cov6), and counts are used directly. No GT.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class NativeSurfelConfig:
    max_iterations: int = 15
    max_correspondence_m: float = 1.5
    min_correspondences: int = 70
    max_scan_points: int = 4000
    min_surfel_count: int = 6
    max_planarity: float = 0.20
    min_horizontal_normal: float = 0.25
    huber_m: float = 0.25
    max_step_xy_m: float = 0.3
    max_step_yaw_deg: float = 0.8
    max_total_xy_m: float = 1.0
    max_total_yaw_deg: float = 4.0
    min_yaw_information: float = 0.001
    candidate_offsets_deg: tuple[float,...] = (0.0,)

@dataclass
class NativeSurfelResult:
    T_target_source: np.ndarray
    converged: bool
    inliers: int
    runtime_ms: float
    failure_reason: str
    iterations: int = 0
    score: float = float('inf')
    yaw_information: float = 0.0


def _surfel_normals(cov6, counts, cfg):
    c = np.asarray(cov6, dtype=np.float64)
    C=np.empty((len(c),3,3),dtype=np.float64)
    C[:,0,0]=c[:,0];C[:,0,1]=C[:,1,0]=c[:,1];C[:,0,2]=C[:,2,0]=c[:,2]
    C[:,1,1]=c[:,3];C[:,1,2]=C[:,2,1]=c[:,4];C[:,2,2]=c[:,5]
    e,v=np.linalg.eigh(C)
    n=v[:,:,0]
    curv=np.maximum(e[:,0],0)/np.maximum(np.sum(np.maximum(e,0),axis=1),1e-12)
    valid=(np.asarray(counts)>=cfg.min_surfel_count)&(curv<=cfg.max_planarity)&(np.hypot(n[:,0],n[:,1])>=cfg.min_horizontal_normal)
    return n,valid, np.sqrt(np.maximum(e[:,0],0))


def _rz(yaw):
    c=math.cos(yaw);s=math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)


def _evaluate(tree, xyz, normals, valid, sigmas, source, T, cfg):
    world=source@T[:3,:3].T+T[:3,3]
    dist,idx=tree.query(world,k=1,workers=1)
    keep=np.isfinite(dist)&(dist<=cfg.max_correspondence_m)&valid[idx]
    if int(keep.sum())<cfg.min_correspondences:
        return None
    indices=np.flatnonzero(keep)
    p=world[indices]; n=normals[idx[indices]]; target=xyz[idx[indices]]
    r=np.einsum('ij,ij->i',n,p-target)
    # Shared robust scale; intrinsic surfel noise is gently regularized, not treated as exact uncertainty.
    scale=np.sqrt(cfg.huber_m**2+sigmas[idx[indices]]**2)
    w=np.minimum(1.0,scale/np.maximum(np.abs(r),1e-9))
    # Robust mean loss normalized per accepted correspondence prevents dense maps winning by volume.
    loss=np.where(np.abs(r)<=scale,0.5*r*r,scale*(np.abs(r)-0.5*scale))
    return indices, p,n,r,w, float(np.mean(loss)),int(keep.sum())


def _solve_single(tree,xyz,normals,valid,sigmas,source,T0,cfg,yaw_offset):
    base=T0.copy()
    # Left-multiplication about the query's own global origin: orientation changes, not XY position.
    R0=_rz(math.radians(yaw_offset))@base[:3,:3]
    T=base.copy();T[:3,:3]=R0
    steps=0; converged=False; reason='max_iterations'; yaw_info=0.0
    for it in range(cfg.max_iterations):
        matched=_evaluate(tree,xyz,normals,valid,sigmas,source,T,cfg)
        if matched is None:
            reason='insufficient_valid_correspondences';break
        ids,p,n,r,w,score,ni=matched
        # left perturbation about current scanner origin, keeping height/roll/pitch fixed
        relative=p-T[:3,3]
        J=np.column_stack((n[:,0],n[:,1], n[:,1]*relative[:,0]-n[:,0]*relative[:,1]))
        sw=np.sqrt(w);A=J*sw[:,None];b=r*sw
        H=A.T@A; g=A.T@b
        Htt=H[:2,:2]+1e-6*np.eye(2)
        yaw_info=float(max(H[2,2]-H[2,:2]@np.linalg.solve(Htt,H[:2,2]),0.0))/len(ids)
        if yaw_info < cfg.min_yaw_information:
            reason='insufficient_yaw_information';break
        try: delta=-np.linalg.solve(H+np.diag([1e-3,1e-3,1e-2]),g)
        except np.linalg.LinAlgError:
            reason='singular_normal_matrix';break
        trans=np.linalg.norm(delta[:2]); lim=cfg.max_step_xy_m
        if trans>lim:delta[:2]*=lim/trans
        delta[2]=float(np.clip(delta[2],-math.radians(cfg.max_step_yaw_deg),math.radians(cfg.max_step_yaw_deg)))
        trial=T.copy();trial[:3,3][:2]+=delta[:2];trial[:3,:3]=_rz(delta[2])@T[:3,:3]
        yaw_delta=math.atan2((trial[:3,:3]@T0[:3,:3].T)[1,0],(trial[:3,:3]@T0[:3,:3].T)[0,0])
        if np.linalg.norm(trial[:2,3]-T0[:2,3])>cfg.max_total_xy_m or abs(yaw_delta)>math.radians(cfg.max_total_yaw_deg):
            reason='update_exceeds_trust_region';break
        new=_evaluate(tree,xyz,normals,valid,sigmas,source,trial,cfg)
        if new is None or new[5]>score+1e-9:
            reason='no_descent';break
        T=trial;steps=it+1
        if np.linalg.norm(delta[:2])<0.005 and abs(delta[2])<math.radians(0.025):
            converged=True;reason='';break
    final=_evaluate(tree,xyz,normals,valid,sigmas,source,T,cfg)
    if final is None: return T,False,0,float('inf'),reason,steps,yaw_info
    # Trustworthy improvement requires geometric descent, regardless of step tolerance.
    initial=T0.copy();initial[:3,:3]=R0
    first=_evaluate(tree,xyz,normals,valid,sigmas,source,initial,cfg)
    improved=first is not None and final[5]<=first[5]+1e-9 and steps>0
    accepted=improved and reason not in ('insufficient_yaw_information','update_exceeds_trust_region','singular_normal_matrix')
    return T,accepted,final[6],final[5],('' if accepted else reason),steps,yaw_info


def align_native_surfels(centroids,cov6,counts,source,T_init,config=None,*,prepared=None):
    cfg=config or NativeSurfelConfig()
    start=time.perf_counter_ns()
    xyz=np.asarray(centroids,dtype=np.float64)
    q=np.asarray(source,dtype=np.float64)
    T0=np.asarray(T_init,dtype=np.float64)
    if len(xyz)<cfg.min_correspondences or len(q)<cfg.min_correspondences:
        return NativeSurfelResult(T0.copy(),False,0,(time.perf_counter_ns()-start)/1e6,'too_few_points')
    # Deterministic stride sampling (independent of map type), not random sampling.
    if len(q)>cfg.max_scan_points:q=q[np.linspace(0,len(q)-1,cfg.max_scan_points,dtype=np.int64)]
    if prepared is None:
        normals,valid,sigmas=_surfel_normals(np.asarray(cov6),np.asarray(counts),cfg)
    else:
        normals,valid,sigmas=prepared
        normals=np.asarray(normals);valid=np.asarray(valid);sigmas=np.asarray(sigmas)
        if normals.shape!=(len(xyz),3) or valid.shape!=(len(xyz),) or sigmas.shape!=(len(xyz),):
            raise ValueError("prepared surfel arrays do not match local map")
    if np.count_nonzero(valid)<cfg.min_correspondences:
        return NativeSurfelResult(T0.copy(),False,0,(time.perf_counter_ns()-start)/1e6,'too_few_valid_surfels')
    tree=cKDTree(xyz)
    results=[_solve_single(tree,xyz,normals,valid,sigmas,q,T0,cfg,d) for d in cfg.candidate_offsets_deg]
    eligible=[r for r in results if r[1]]
    if not eligible:
        best=min(results,key=lambda r:r[3]);return NativeSurfelResult(T0.copy(),False,best[2],(time.perf_counter_ns()-start)/1e6,best[4],best[5],best[3],best[6])
    best=min(eligible,key=lambda r:(r[3],-r[2]))
    return NativeSurfelResult(best[0],True,best[2],(time.perf_counter_ns()-start)/1e6,'',best[5],best[3],best[6])
