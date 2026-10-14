import numpy as np
from scipy.spatial.transform import Rotation
from localization.bev_local_pose import planar_rigid,estimate_pose
from localization.surfel_6dof import align_surfel_6dof,Surfel6DConfig

def test_planar_svd_and_local_match():
    rng=np.random.default_rng(5);q=rng.uniform(-20,20,(90,2));a=0.32
    R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    r=q@R.T+np.array([2.,-1.])
    feats=np.eye(90,dtype=np.float32)
    fit=estimate_pose(q,feats,r,feats,min_inliers=30,max_ransac=120)
    assert fit.converged,fit.failure_reason
    assert np.allclose(fit.T_target_source[:2,:2],R,atol=1e-5)
    assert np.allclose(fit.T_target_source[:2,3],[2.,-1.],atol=1e-5)

def test_six_dof_surfel_refinement():
    rng=np.random.default_rng(12)
    n=600
    # Three orthogonal wall/ground families, with well-conditioned normals.
    x=rng.uniform(-10,10,(n//3,2));y=rng.uniform(-10,10,(n//3,2));z=rng.uniform(-10,10,(n//3,2))
    centers=np.concatenate([np.column_stack((x,np.zeros(n//3))),
        np.column_stack((np.zeros(n//3),y)),
        np.column_stack((z[:,0],np.zeros(n//3),z[:,1]))])
    normals=np.concatenate([np.tile([0,0,1],(n//3,1)),np.tile([1,0,0],(n//3,1)),np.tile([0,1,0],(n//3,1))])
    cov=np.zeros((n,6));cov[:,0]=.01;cov[:,3]=.01;cov[:,5]=.01
    for i,k in enumerate(normals):
        C=.01*np.eye(3)-.0099*np.outer(k,k)
        cov[i]=[C[0,0],C[0,1],C[0,2],C[1,1],C[1,2],C[2,2]]
    counts=np.full(n,20)
    source=centers.copy()
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.005,-.007,.01]).as_matrix();T[:3,3]=[.06,-.06,.07]
    res=align_surfel_6dof(centers,cov,counts,source,T,Surfel6DConfig(min_inliers=60))
    assert res.converged,res.failure_reason
    assert np.linalg.norm(res.T_target_source[:3,3])<np.linalg.norm(T[:3,3])
    assert Rotation.from_matrix(res.T_target_source[:3,:3]).magnitude()<Rotation.from_matrix(T[:3,:3]).magnitude()
