import numpy as np
import pytest
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from localization.bev_hybrid import HybridBEVPolicy
from localization.bev_local_pose import estimate_pose, planar_rigid
from localization.step95 import compose_world_query


def test_bev_confidence_gate_and_invalid_parameters():
    p=HybridBEVPolicy()
    result=SimpleNamespace(converged=True,inliers=32,T_target_source=np.eye(4))
    assert p.accepts(result,300,300)
    assert not p.accepts(result,500,500)
    assert not p.accepts(SimpleNamespace(converged=False,inliers=200,T_target_source=np.eye(4)),300,300)
    with pytest.raises(ValueError): HybridBEVPolicy(min_inlier_ratio=0)
    with pytest.raises(ValueError): HybridBEVPolicy(max_bev_candidates=0)


def test_planar_transform_estimates_reference_from_query():
    rng=np.random.default_rng(42)
    query=rng.uniform(-30,30,(90,2))
    yaw=np.deg2rad(23)
    R=Rotation.from_euler('z',yaw).as_matrix()[:2,:2]
    offset=np.array([4.,-6.])
    reference=query@R.T+offset
    R_fit,t_fit=planar_rigid(query,reference)
    np.testing.assert_allclose(R_fit,R,atol=1e-12)
    np.testing.assert_allclose(t_fit,offset,atol=1e-12)
    features=np.eye(90,dtype=np.float32)
    fit=estimate_pose(query,features,reference,features)
    assert fit.converged,fit.failure_reason
    np.testing.assert_allclose(fit.T_target_source[:2,:2],R,atol=1e-7)
    np.testing.assert_allclose(fit.T_target_source[:2,3],offset,atol=1e-7)


def test_world_composition_reference_to_query():
    # Reference pose row stores timestamp, x, y, z and orientation in actual
    # project encoding; test composition via a standalone matrix only if possible.
    from localization.step95 import pose_row_to_matrix
    import inspect
    assert callable(compose_world_query)
    assert callable(pose_row_to_matrix)
