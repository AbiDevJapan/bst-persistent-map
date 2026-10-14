"""Step 9.7D conservative early exit for ordered LiDAR candidates.

Only a converged scan-GICP with sufficient inlier support can terminate
candidate search. This is a heuristic; it is NOT a validated correctness
certificate. In particular, periodic environments may accept a false place.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EarlyExitPolicy:
    min_inliers: int = 350
    min_inlier_fraction: float = 0.025

    def __post_init__(self):
        if self.min_inliers < 1 or not math.isfinite(self.min_inlier_fraction) or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError('Invalid early-exit threshold')


def verified_candidate(kiss_result, scan_gicp_result, query_point_count, policy=EarlyExitPolicy()):
    """Cheap provisional geometric verification, independent of GT pose.

    Do not use descriptor confidence to override failed geometry. The gate
    deliberately requires scan-GICP convergence; this still does not rule out
    perceptual aliasing. Benchmark global failures before deployment.
    """
    if scan_gicp_result is None or kiss_result is None or query_point_count < 1:
        return False
    if not bool(kiss_result.converged) or not bool(scan_gicp_result.converged):
        return False
    inliers = int(scan_gicp_result.inliers)
    return inliers >= policy.min_inliers and inliers/query_point_count >= policy.min_inlier_fraction


@dataclass(frozen=True)
class GeometryVerificationPolicy:
    """Independent, ground-truth-free geometric check for adaptive early exit.

    `max_points` caps *query* validation work (stable regularly-spaced subsample).
    The reference tree uses all passed, preprocessed reference points. This check
    only determines EARLY STOP; exhaustive candidates retain their old scoring.
    """
    overlap_radius_m: float = 1.0
    min_overlap: float = 0.55
    max_trimmed_rmse_m: float = 0.65
    trim_fraction: float = 0.8
    max_points: int = 2500

    def __post_init__(self):
        vals=(self.overlap_radius_m,self.min_overlap,self.max_trimmed_rmse_m,self.trim_fraction)
        if not all(math.isfinite(x) for x in vals) or self.overlap_radius_m <= 0 or self.max_trimmed_rmse_m <= 0:
            raise ValueError('Invalid geometric verification distances')
        if not 0 < self.min_overlap <= 1 or not 0 < self.trim_fraction <= 1 or self.max_points < 32:
            raise ValueError('Invalid geometric verification thresholds')


def verify_alignment_geometry(query_points, reference_points, T_reference_query, policy=GeometryVerificationPolicy()):
    """Return (accepted, overlap, trimmed_rmse) with independent NN support.

    Scan points are preprocessed in the same local sensor frame as global matching.
    No ground truth, historical reference poses, or subsequent map refinement used.
    Degenerate/invalid inputs fail closed. Registration correspondences/inliers are
    NOT reused to avoid simply duplicating GICP convergence statistics.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    q=np.asarray(query_points,dtype=np.float64)
    r=np.asarray(reference_points,dtype=np.float64)
    T=np.asarray(T_reference_query,dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] < 3 or r.shape[1] < 3 or len(q)<32 or len(r)<32 or T.shape!=(4,4):
        return False,0.0,float('inf')
    if not (np.isfinite(q[:,:3]).all() and np.isfinite(r[:,:3]).all() and np.isfinite(T).all()):
        return False,0.0,float('inf')
    if len(q)>policy.max_points:
        indices=np.linspace(0,len(q)-1,policy.max_points,dtype=np.int64)
        q=q[indices]
    warped=q[:,:3]@T[:3,:3].T+T[:3,3]
    d,_=cKDTree(r[:,:3]).query(warped,k=1,workers=1)
    within=d<=policy.overlap_radius_m
    overlap=float(np.mean(within))
    if not np.any(within):
        return False,overlap,float('inf')
    ds=np.sort(d[within]);k=max(1,int(math.ceil(policy.trim_fraction*len(ds))))
    rmse=float(np.sqrt(np.mean(ds[:k]**2)))
    return bool(overlap>=policy.min_overlap and rmse<=policy.max_trimmed_rmse_m),overlap,rmse
