import numpy as np
import pytest
from localization.native_surfel_prepared import PreparedSurfelMap
from localization.native_surfel_registration import NativeSurfelConfig, align_native_surfels
from tests.test_native_surfel_registration import scene


def test_prepared_replays_identically():
    p,c,n=scene()
    cfg=NativeSurfelConfig(min_correspondences=35,max_scan_points=1000,max_correspondence_m=1.5,min_yaw_information=1e-5)
    T=np.eye(4);a=np.deg2rad(1.4)
    T[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
    T[0,3]=0.12
    original=align_native_surfels(p,c,n,p,T,cfg)
    cache=PreparedSurfelMap.build(c,n,cfg,chunk_size=17)
    optim=align_native_surfels(p,c,n,p,T,cfg,prepared=cache.select(np.arange(len(p))))
    assert np.array_equal(original.T_target_source,optim.T_target_source)
    assert original.converged==optim.converged
    assert original.failure_reason==optim.failure_reason
    assert original.inliers==optim.inliers
    assert original.score==optim.score


def test_selected_order_and_shapes():
    p,c,n=scene()
    cache=PreparedSurfelMap.build(c,n,chunk_size=23)
    ids=np.array([12,2,12,0])
    normals,valid,sigmas=cache.select(ids)
    assert normals.shape==(4,3)
    assert np.array_equal(normals[0],normals[2])
    with pytest.raises(IndexError):cache.select([-1])
    with pytest.raises(ValueError): align_native_surfels(p,c,n,p,np.eye(4),prepared=(np.zeros((2,3)),np.zeros(2),np.zeros(2)))
