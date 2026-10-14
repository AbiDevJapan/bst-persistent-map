import numpy as np
from localization.native_surfel_registration import NativeSurfelConfig, align_native_surfels

def scene():
    # Perpendicular vertical walls (yaw observable), with explicit sufficient-statistic surfels.
    a=np.linspace(-8,8,33);z=np.linspace(-1,2,5)
    p=np.concatenate([np.array([(0,y,zz) for y in a for zz in z]),np.array([(x,6,zz) for x in a for zz in z])],axis=0)
    cov=np.zeros((len(p),6)); cov[:,0]=.02;cov[:,3]=.02;cov[:,5]=.02
    cov[:len(a)*len(z),0]=.0002
    cov[len(a)*len(z):,3]=.0002
    return p,cov,np.full(len(p),20,dtype=int)

def test_identity_and_wrong_heading():
    p,c,n=scene(); cfg=NativeSurfelConfig(min_correspondences=35,max_scan_points=1000,max_correspondence_m=1.5,min_yaw_information=1e-5)
    theta=np.deg2rad(1.4);T=np.eye(4);T[:3,:3]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    T[0,3]=.12
    r=align_native_surfels(p,c,n,p,T,cfg)
    assert np.all(np.isfinite(r.T_target_source))
    assert r.runtime_ms>=0
    if r.converged:
        assert abs(np.arctan2(r.T_target_source[1,0],r.T_target_source[0,0]))<=abs(theta)+1e-5

def test_invalid_target_falls_back():
    p,c,n=scene();T=np.eye(4)
    r=align_native_surfels(p[:10],c[:10],n[:10],p,T)
    assert not r.converged and np.array_equal(r.T_target_source,T)

def test_no_randomness():
    p,c,n=scene();T=np.eye(4)
    x=align_native_surfels(p,c,n,p,T)
    y=align_native_surfels(p,c,n,p,T)
    assert np.allclose(x.T_target_source,y.T_target_source)
