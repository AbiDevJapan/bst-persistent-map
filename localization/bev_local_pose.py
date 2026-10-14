"""Step 9.7F: BEVPlace++ local feature matching, planar rigid pose.

Input is FAST keypoints and REIN local descriptors from the official checkpoint.
Coordinates follow upstream NCLT 201x201 BEV: x=(100-row)*0.4,
y=(100-col)*0.4; output is reference<-query (SE2 embedded SE3).
"""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class BEVLocalResult:
    T_target_source: np.ndarray
    converged: bool
    inliers: int
    runtime_ms: float
    failure_reason: str = ''


def planar_rigid(src,dst):
    src=np.asarray(src,dtype=float);dst=np.asarray(dst,dtype=float)
    if src.shape!=dst.shape or src.ndim!=2 or src.shape[1]!=2 or len(src)<2:raise ValueError('Need paired xy points')
    a=src-src.mean(0);b=dst-dst.mean(0)
    U,_,Vt=np.linalg.svd(a.T@b)
    S=np.eye(2);S[-1,-1]=np.linalg.det(Vt.T@U.T)
    R=Vt.T@S@U.T
    t=dst.mean(0)-R@src.mean(0)
    return R,t


def estimate_pose(q_xy,q_desc,r_xy,r_desc,min_matches=14,min_inliers=12,max_ransac=600,threshold_m=0.6):
    t0=time.perf_counter()
    def res(T,ok,n,why=''):
        return BEVLocalResult(T,ok,n,(time.perf_counter()-t0)*1000,why)
    identity=np.eye(4)
    q=np.asarray(q_xy,dtype=float);r=np.asarray(r_xy,dtype=float)
    a=np.asarray(q_desc,dtype=np.float32);b=np.asarray(r_desc,dtype=np.float32)
    if q.ndim!=2 or r.ndim!=2 or q.shape[1:]!=(2,) or r.shape[1:]!=(2,) or a.shape!=(len(q),a.shape[-1]) or b.shape!=(len(r),a.shape[-1]):
        raise ValueError('Invalid BEV local features')
    if len(q)<min_matches or len(r)<min_matches:
        return res(identity,False,0,'insufficient_features')
    # Descriptor L2 normalized upstream for stable cross-scan nearest-neighbors.
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
    b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8)
    dist,inds=cKDTree(b).query(a,k=2,workers=1)
    good=(dist[:,0]<0.88*dist[:,1])
    qq=q[good];rr=r[inds[good,0]]
    if len(qq)<min_matches:
        return res(identity,False,len(qq),'insufficient_ratio_matches')
    # Deterministic two-point RANSAC. No ground truth, no learned pose labels.
    rng=np.random.default_rng(97);best=None;best_count=0
    n=len(qq)
    for _ in range(max_ransac):
        i,j=rng.choice(n,2,replace=False)
        if np.linalg.norm(qq[i]-qq[j])<1.0 or np.linalg.norm(rr[i]-rr[j])<1.0:continue
        R,t=planar_rigid(qq[[i,j]],rr[[i,j]])
        error=np.linalg.norm(qq@R.T+t-rr,axis=1)
        mask=error<threshold_m
        if int(mask.sum())>best_count:
            best=mask;best_count=int(mask.sum())
    if best is None or best_count<min_inliers:
        return res(identity,False,best_count,'ransac_insufficient_inliers')
    R,t=planar_rigid(qq[best],rr[best])
    # Reject geometrically unstable tight clusters.
    if np.linalg.svd(qq[best]-qq[best].mean(0),compute_uv=False)[-1]<0.5:
        return res(identity,False,best_count,'degenerate_feature_geometry')
    T=np.eye(4);T[:2,:2]=R;T[:2,3]=t
    return res(T,True,best_count)

class LocalFeatureStore:
    """One NPZ per historical session; row indices identical to descriptor cache."""
    def __init__(self,folder):
        self.folder=Path(folder);self.loaded={}
    def get(self,kind,session,index,scan_file):
        key=(kind,session)
        if key not in self.loaded:
            filename=self.folder/f'{kind}_{session}_local.npz'
            if not filename.is_file():raise FileNotFoundError(filename)
            z=np.load(filename,allow_pickle=False)
            self.loaded[key]={k:z[k] for k in z.files}
            z.close()
        z=self.loaded[key];i=int(index)
        if str(z['scan_files'][i])!=str(scan_file):raise ValueError(f'feature cache row mismatch for {key}, {i}')
        lo=int(z['offsets'][i]);hi=int(z['offsets'][i+1])
        return z['xy'][lo:hi],z['features'][lo:hi]
