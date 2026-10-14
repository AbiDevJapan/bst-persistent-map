"""Step 9.7F: robust 6-DoF point-to-native-surfel registration.

Uses frozen map normals/centroids; damped point-to-plane Gauss Newton in SE(3).
No GT, and no height/roll/pitch constraints derived from GT.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from localization.native_surfel_registration import _surfel_normals, NativeSurfelConfig, NativeSurfelResult

@dataclass(frozen=True)
class Surfel6DConfig:
    iterations: int = 18
    correspondence_m: float = 1.5
    max_points: int = 4000
    min_inliers: int = 100
    huber_m: float = 0.20
    max_step_m: float = 0.25
    max_step_deg: float = 1.5
    max_total_translation_m: float = 1.8
    max_total_rotation_deg: float = 8.0
    damping: float = 1e-3
    profile: bool = False


def align_surfel_6dof(centroids, cov6, counts, scan, initial, config=None, prepared=None, fast_tree=False):
    cfg=config or Surfel6DConfig()
    t0=time.perf_counter()
    phases={"normal_ms":0.0,"tree_ms":0.0,"correspondence_ms":0.0,"linearize_ms":0.0,"acceptance_ms":0.0}
    T0=np.asarray(initial,dtype=np.float64)
    def result(T,success,n,reason,it=0,score=float('inf')):
        out=NativeSurfelResult(T,success,n,(time.perf_counter()-t0)*1000,reason,it,score)
        out.profile_ms=dict(phases)
        return out
    dst=np.asarray(centroids,dtype=np.float64)
    src=np.asarray(scan,dtype=np.float64)
    if T0.shape!=(4,4) or not np.isfinite(T0).all() or dst.ndim!=2 or src.ndim!=2 or dst.shape[1:]!=(3,) or src.shape[1:]!=(3,):
        raise ValueError('Invalid SE3 or point array')
    if len(dst)<cfg.min_inliers or len(src)<cfg.min_inliers:
        return result(T0.copy(),False,0,'insufficient_points')
    if len(src)>cfg.max_points:
        src=src[np.linspace(0,len(src)-1,cfg.max_points).astype(np.int64)]
    tick=time.perf_counter()
    if prepared is None:
        n,valid,_=_surfel_normals(cov6,counts,NativeSurfelConfig(min_horizontal_normal=0.0))
    else:
        n,valid,_=prepared
        if len(n)!=len(dst) or len(valid)!=len(dst):
            raise ValueError("Prepared surfel arrays do not match selected map")
        # The 6-DoF prepared cache is built with min_horizontal_normal=0.0.
        # Do not reuse the XY+yaw validity mask, which rejects ground surfels.
    phases["normal_ms"]+=(time.perf_counter()-tick)*1000
    if not np.isfinite(dst).all() or not np.isfinite(n).all():
        raise ValueError('Nonfinite target geometry')
    tick=time.perf_counter(); tree=(cKDTree(dst, balanced_tree=False, compact_nodes=False)
          if fast_tree else cKDTree(dst))
    phases["tree_ms"]+=(time.perf_counter()-tick)*1000
    T=T0.copy(); prev=float('inf'); accepted=False; count=0; steps=0
    for iteration in range(cfg.iterations):
        tick=time.perf_counter()
        p=src@T[:3,:3].T+T[:3,3]
        dist,idx=tree.query(p,k=1,workers=1)
        mask=(dist<=cfg.correspondence_m)&valid[idx]
        count=int(mask.sum())
        phases["correspondence_ms"]+=(time.perf_counter()-tick)*1000
        if count<cfg.min_inliers:
            return result(T0.copy(),False,count,'insufficient_correspondences',iteration)
        tick=time.perf_counter()
        q=p[mask]; target=dst[idx[mask]]; normal=n[idx[mask]]
        residual=np.einsum('ij,ij->i',normal,q-target)
        weights=np.minimum(1.0,cfg.huber_m/np.maximum(np.abs(residual),1e-9))
        J=np.concatenate((normal,np.cross(q-T[:3,3],normal)),axis=1)
        a=J*np.sqrt(weights)[:,None]; b=residual*np.sqrt(weights)
        H=a.T@a; g=a.T@b
        # Check conditioning before claiming a successful six-axis pose.
        eig=np.linalg.eigvalsh(H)
        if eig[0] <= max(1e-7, eig[-1]*1e-7):
            return result(T0.copy(),False,count,'degenerate_6d_geometry',iteration)
        delta=-np.linalg.solve(H+cfg.damping*np.diag(np.maximum(np.diag(H),1.0)),g)
        phases["linearize_ms"]+=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        tr=delta[:3]; rot=delta[3:]
        nt=np.linalg.norm(tr);nr=np.linalg.norm(rot)
        if nt>cfg.max_step_m:tr*=cfg.max_step_m/nt
        cap=np.deg2rad(cfg.max_step_deg)
        if nr>cap:rot*=cap/nr
        Rdelta=Rotation.from_rotvec(rot).as_matrix()
        candidate=T.copy()
        candidate[:3,:3]=Rdelta@T[:3,:3]
        candidate[:3,3]=T[:3,3]+tr
        delta_xyz=np.linalg.norm(candidate[:3,3]-T0[:3,3]);delta_angle=Rotation.from_matrix(candidate[:3,:3]@T0[:3,:3].T).magnitude()
        if delta_xyz>cfg.max_total_translation_m or delta_angle>np.deg2rad(cfg.max_total_rotation_deg):
            return result(T0.copy(),False,count,'trust_region_exceeded',iteration)
        # Compare robust objective on identical original correspondences, preventing
        # a NN assignment change from falsely appearing to improve objective.
        newp=src[mask]@candidate[:3,:3].T+candidate[:3,3]
        rn=np.einsum('ij,ij->i',normal,newp-target)
        def huber(r):
            ar=np.abs(r);return np.mean(np.where(ar<=cfg.huber_m,0.5*r*r,cfg.huber_m*(ar-0.5*cfg.huber_m)))
        old=huber(residual);new=huber(rn)
        phases["acceptance_ms"]+=(time.perf_counter()-tick)*1000
        if new>old+1e-12:
            return result(T if accepted else T0.copy(),accepted,count,'line_search_rejected',iteration,prev)
        T=candidate;accepted=True;steps=iteration+1;prev=new
        if np.linalg.norm(tr)<1e-4 and np.linalg.norm(rot)<2e-5:
            break
    return result(T,accepted,count,'' if accepted else 'no_accepted_step',steps,prev)
