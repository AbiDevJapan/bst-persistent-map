import numpy as np
from localization.native_surfel_prepared import PreparedSurfelMap
from localization.native_surfel_registration import NativeSurfelConfig
from localization.surfel_6dof import align_surfel_6dof,Surfel6DConfig

def test_prepared_matches_unprepared_full_6dof():
    rng=np.random.default_rng(99); n=600
    xyz=rng.uniform(-12,12,(n,3))
    normals=np.eye(3)[np.arange(n)%3]
    cov=np.empty((n,6))
    for i,normal in enumerate(normals):
        C=np.eye(3)*0.01-0.0099*np.outer(normal,normal)
        cov[i]=[C[0,0],C[0,1],C[0,2],C[1,1],C[1,2],C[2,2]]
    counts=np.full(n,20)
    prep=PreparedSurfelMap.build(cov,counts,NativeSurfelConfig(min_horizontal_normal=0.0))
    assert np.any(prep.valid[::3]) # Ground normals remain valid
    T=np.eye(4);T[:3,3]=[0.03,-0.04,0.05]
    cfg=Surfel6DConfig(min_inliers=60,iterations=5)
    a=align_surfel_6dof(xyz,cov,counts,xyz,T,cfg)
    b=align_surfel_6dof(xyz,cov,counts,xyz,T,cfg,prepared=prep.select(np.arange(n)))
    assert a.failure_reason==b.failure_reason
    assert a.converged==b.converged
    assert a.inliers==b.inliers
    assert a.iterations==b.iterations
    np.testing.assert_allclose(a.T_target_source,b.T_target_source,atol=1e-12,rtol=0)
    assert b.profile_ms["normal_ms"]>=0
    assert b.profile_ms["tree_ms"]>=0
    assert b.profile_ms["correspondence_ms"]>=0

def test_prepared_size_check():
    centers=np.zeros((120,3));cov=np.zeros((120,6));cnt=np.ones(120)
    try:
        align_surfel_6dof(centers,cov,cnt,centers,np.eye(4),prepared=(np.zeros((119,3)),np.ones(119,bool),np.zeros(119)))
    except ValueError as exc:
        assert 'Prepared surfel arrays' in str(exc)
    else:
        raise AssertionError('Expected mismatch rejection')
