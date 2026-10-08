from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from localization.step90 import Pose3DOF, level_body_points, transform_leveled, wrap_angle
from localization.step901 import PointToPlaneICP3DOFConfig, PointToPlaneICP3DOFResult, _limit_increment
from localization.step902 import PyramidLevelConfig, DEFAULT_PYRAMID, voxel_downsample_first
from realdata.nclt_packed import linear_decode

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


class SurfelLocalMapIndex:
    """Compact bucket index returning base-surfel row IDs, not voxel centers."""
    def __init__(self, surfel_map, bucket_width_m: float = 8.0):
        self.map = surfel_map
        self.codes = np.asarray(surfel_map.linear_codes, dtype=np.uint64)
        self.spec = surfel_map.spec
        self.origin = np.asarray(surfel_map.origin_local_m, dtype=np.float64)
        self.bucket_voxels = max(1, int(round(float(bucket_width_m) / self.spec.voxel_size_m)))
        x, y, _ = linear_decode(self.codes, self.spec)
        bx = x.astype(np.int64) // self.bucket_voxels
        by = y.astype(np.int64) // self.bucket_voxels
        self.nby = int(math.ceil(self.spec.ny / self.bucket_voxels))
        key = bx * self.nby + by
        order = np.argsort(key, kind="stable")
        self.order = order.astype(np.uint32 if len(order) < 2**32 else np.uint64)
        ks = key[order]
        self.bucket_keys, self.bucket_starts, self.bucket_counts = np.unique(ks, return_index=True, return_counts=True)

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.order, self.bucket_keys, self.bucket_starts, self.bucket_counts)))

    def query_radius_ids(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        vs = self.spec.voxel_size_m
        gx0 = max(0, int(math.floor((cx-radius_m-self.origin[0])/vs)))
        gx1 = min(self.spec.nx, int(math.ceil((cx+radius_m-self.origin[0])/vs)))
        gy0 = max(0, int(math.floor((cy-radius_m-self.origin[1])/vs)))
        gy1 = min(self.spec.ny, int(math.ceil((cy+radius_m-self.origin[1])/vs)))
        if gx0 >= gx1 or gy0 >= gy1:
            return np.empty(0, dtype=np.int64)
        bx0, bx1 = gx0//self.bucket_voxels, (gx1-1)//self.bucket_voxels
        by0, by1 = gy0//self.bucket_voxels, (gy1-1)//self.bucket_voxels
        parts=[]
        for bx in range(bx0,bx1+1):
            for by in range(by0,by1+1):
                k=bx*self.nby+by
                j=int(np.searchsorted(self.bucket_keys,k))
                if j < len(self.bucket_keys) and int(self.bucket_keys[j]) == k:
                    a=int(self.bucket_starts[j]); b=a+int(self.bucket_counts[j])
                    parts.append(self.order[a:b])
        if not parts:
            return np.empty(0,dtype=np.int64)
        ids=np.concatenate(parts).astype(np.int64,copy=False) if len(parts)>1 else parts[0].astype(np.int64,copy=False)
        c=self.map.centroids[ids]
        r2=(c[:,0]-cx)**2+(c[:,1]-cy)**2
        return ids[r2 <= radius_m*radius_m]


@dataclass
class PreparedSurfelLevel:
    config: PyramidLevelConfig
    centroids: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    tree: object


@dataclass
class CoarseToFineSurfelResult:
    pose: Pose3DOF
    converged: bool
    iterations: int
    correspondences: int
    normal_correspondences: int
    rmse_m: float
    mean_abs_plane_residual_m: float
    match_ms: float
    failure_reason: str
    level_results: list[tuple[str, PointToPlaneICP3DOFResult]]


def _cov6_to_matrix(c6: np.ndarray) -> np.ndarray:
    a=np.asarray(c6,dtype=np.float64)
    C=np.empty((len(a),3,3),dtype=np.float64)
    C[:,0,0]=a[:,0]; C[:,0,1]=a[:,1]; C[:,0,2]=a[:,2]
    C[:,1,0]=a[:,1]; C[:,1,1]=a[:,3]; C[:,1,2]=a[:,4]
    C[:,2,0]=a[:,2]; C[:,2,1]=a[:,4]; C[:,2,2]=a[:,5]
    return C


def _matrix_to_cov6(C: np.ndarray) -> np.ndarray:
    return np.column_stack((C[:,0,0],C[:,0,1],C[:,0,2],C[:,1,1],C[:,1,2],C[:,2,2]))


def aggregate_surfels(
    centroids: np.ndarray,
    cov6: np.ndarray,
    counts: np.ndarray,
    voxel_m: float,
    normal_min_points: int,
    normal_max_curvature: float,
    normal_min_horizontal_component: float,
):
    mu=np.asarray(centroids,dtype=np.float64)
    cv=np.asarray(cov6,dtype=np.float64)
    w=np.asarray(counts,dtype=np.float64)
    if not len(mu):
        return (np.empty((0,3)),np.empty((0,3)),np.empty(0,dtype=bool),np.empty(0,dtype=np.uint32),np.empty((0,6)))
    keys=np.floor(mu/float(voxel_m)).astype(np.int64)
    order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]))
    keys=keys[order]; mu=mu[order]; cv=cv[order]; w=w[order]
    starts=np.r_[0,np.flatnonzero(np.any(keys[1:] != keys[:-1],axis=1))+1]
    W=np.add.reduceat(w,starts)
    msum=np.add.reduceat(mu*w[:,None],starts,axis=0)
    M=msum/W[:,None]
    C=_cov6_to_matrix(cv)
    outer=mu[:,:,None]*mu[:,None,:]
    second=(C+outer)*w[:,None,None]
    S=np.add.reduceat(second,starts,axis=0)/W[:,None,None]
    AC=S-M[:,:,None]*M[:,None,:]
    ev,evec=np.linalg.eigh(AC)
    total=np.sum(ev,axis=1)
    curv=np.divide(np.maximum(ev[:,0],0.0),total,out=np.full(len(W),np.inf),where=total>1e-12)
    n=evec[:,:,0]
    h=np.hypot(n[:,0],n[:,1])
    valid=(W>=int(normal_min_points)) & np.isfinite(curv) & (curv<=float(normal_max_curvature)) & (h>=float(normal_min_horizontal_component))
    return M,n,valid,W.astype(np.uint32),_matrix_to_cov6(AC)


def prepare_surfel_pyramid(surfel_map, ids: np.ndarray, levels=DEFAULT_PYRAMID, cfg: PointToPlaneICP3DOFConfig|None=None):
    if cKDTree is None:
        raise RuntimeError("Step 9.0.3B requires scipy.spatial.cKDTree") from _SCIPY_IMPORT_ERROR
    cfg0=cfg or PointToPlaneICP3DOFConfig()
    ids=np.asarray(ids,dtype=np.int64)
    c=surfel_map.centroids[ids]
    cv=surfel_map.cov6[ids]
    w=surfel_map.counts[ids]
    out=[]
    for lev in levels:
        mu,n,valid,cnt,_=aggregate_surfels(c,cv,w,lev.map_voxel_m,cfg0.normal_min_neighbors,cfg0.normal_max_curvature,cfg0.normal_min_horizontal_component)
        tree=cKDTree(mu) if len(mu) else None
        out.append(PreparedSurfelLevel(lev,mu,n,valid,cnt,tree))
    return out


def scan_to_surfel_point_to_plane_icp_3dof(points_body,map_points,normals,normal_valid,initial_pose,fixed_z,fixed_roll,fixed_pitch,config=None,map_tree=None):
    if cKDTree is None:
        raise RuntimeError("Step 9.0.3B requires scipy.spatial.cKDTree") from _SCIPY_IMPORT_ERROR
    cfg=config or PointToPlaneICP3DOFConfig()
    pts=np.asarray(points_body,dtype=np.float64); mp=np.asarray(map_points,dtype=np.float64)
    ns_all=np.asarray(normals,dtype=np.float64); valid_all=np.asarray(normal_valid,dtype=bool)
    if len(pts)==0:
        return PointToPlaneICP3DOFResult(initial_pose,False,0,0,float('inf'),0.0,'empty_scan')
    if len(mp)<cfg.min_correspondences:
        return PointToPlaneICP3DOFResult(initial_pose,False,0,0,float('inf'),0.0,'local_map_too_small')
    tree=map_tree if map_tree is not None else cKDTree(mp)
    q=level_body_points(pts,fixed_roll,fixed_pitch)
    pose=initial_pose; conv=False; reason='max_iterations'; lc=ln=0; lr=la=float('inf')
    t0=time.perf_counter_ns()
    for it in range(1,cfg.max_iterations+1):
        world=transform_leveled(q,pose,fixed_z)
        dist,idx=tree.query(world,k=1,workers=1)
        keep=np.isfinite(dist)&(dist<=cfg.max_correspondence_m)&valid_all[idx]
        ids=np.flatnonzero(keep)
        if len(ids)<cfg.min_correspondences:
            reason='too_few_normal_correspondences'; lc=int(np.count_nonzero(np.isfinite(dist)&(dist<=cfg.max_correspondence_m))); ln=int(len(ids)); break
        tidx=idx[ids]; targets=mp[tidx]; ns=ns_all[tidx]
        residual=np.einsum('ij,ij->i',ns,world[ids]-targets)
        if 0<cfg.trim_fraction<1 and len(ids)>cfg.min_correspondences:
            take=max(cfg.min_correspondences,int(math.ceil(len(ids)*cfg.trim_fraction)))
            sel=np.argpartition(np.abs(residual),take-1)[:take]
            ids=ids[sel]; targets=targets[sel]; ns=ns[sel]; residual=residual[sel]
        qq=q[ids]; c0,s0=math.cos(pose.yaw),math.sin(pose.yaw)
        dpx=-s0*qq[:,0]-c0*qq[:,1]; dpy=c0*qq[:,0]-s0*qq[:,1]
        J=np.empty((len(ids),3),dtype=np.float64)
        J[:,0]=ns[:,0]; J[:,1]=ns[:,1]; J[:,2]=ns[:,0]*dpx+ns[:,1]*dpy
        A=J.T@J; b=-(J.T@residual); A.flat[::4]+=float(cfg.damping)
        try: delta=np.linalg.solve(A,b)
        except np.linalg.LinAlgError: delta,*_=np.linalg.lstsq(A,b,rcond=None)
        if not np.all(np.isfinite(delta)):
            reason='nonfinite_update'; break
        delta=_limit_increment(delta,cfg)
        pose=Pose3DOF(pose.x+float(delta[0]),pose.y+float(delta[1]),wrap_angle(pose.yaw+float(delta[2])))
        lc=int(np.count_nonzero(np.isfinite(dist)&(dist<=cfg.max_correspondence_m))); ln=int(len(ids))
        lr=float(math.sqrt(np.mean(residual*residual))); la=float(np.mean(np.abs(residual)))
        if math.hypot(float(delta[0]),float(delta[1]))<=cfg.translation_tolerance_m and abs(math.degrees(float(delta[2])))<=cfg.yaw_tolerance_deg:
            conv=True; reason=''; break
    return PointToPlaneICP3DOFResult(pose,conv,it if 'it' in locals() else 0,lc,lr,(time.perf_counter_ns()-t0)/1e6,reason,ln,la)


def scan_to_map_coarse_to_fine_surfel_3dof(points_body,prepared_levels,initial_pose,fixed_z,fixed_roll,fixed_pitch,base_config=None):
    cfg0=base_config or PointToPlaneICP3DOFConfig(); pose=initial_pose; results=[]; total_it=0; total_ms=0.; final=None
    pts=np.asarray(points_body,dtype=np.float64)
    for p in prepared_levels:
        lev=p.config; scan=voxel_downsample_first(pts,lev.scan_voxel_m)
        cfg=replace(cfg0,max_iterations=int(lev.max_iterations),max_correspondence_m=float(lev.max_correspondence_m),trim_fraction=float(lev.trim_fraction),max_step_translation_m=float(lev.max_step_translation_m),max_step_yaw_deg=float(lev.max_step_yaw_deg))
        res=scan_to_surfel_point_to_plane_icp_3dof(scan,p.centroids,p.normals,p.valid,pose,fixed_z,fixed_roll,fixed_pitch,cfg,p.tree)
        results.append((lev.name,res)); total_it+=res.iterations; total_ms+=res.match_ms; pose=res.pose; final=res
        if res.failure_reason in {'empty_scan','local_map_too_small','too_few_correspondences','too_few_normal_correspondences','nonfinite_update'}: break
    if final is None: final=PointToPlaneICP3DOFResult(initial_pose,False,0,0,float('inf'),0.0,'no_levels')
    return CoarseToFineSurfelResult(pose,bool(final.converged and len(results)==len(prepared_levels)),total_it,final.correspondences,final.normal_correspondences,final.rmse_m,final.mean_abs_plane_residual_m,total_ms,final.failure_reason,results)


__all__=['SurfelLocalMapIndex','PreparedSurfelLevel','CoarseToFineSurfelResult','aggregate_surfels','prepare_surfel_pyramid','scan_to_surfel_point_to_plane_icp_3dof','scan_to_map_coarse_to_fine_surfel_3dof']
