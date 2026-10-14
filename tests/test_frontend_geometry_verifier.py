import numpy as np
import pytest
from localization.frontend_adaptive import GeometryVerificationPolicy, verify_alignment_geometry

def test_perfect_alignment_passes():
    rng=np.random.default_rng(12)
    q=rng.normal(size=(500,3))
    T=np.eye(4);T[:3,3]=[2,-3,0.4]
    ref=q+T[:3,3]
    ok,ov,rmse=verify_alignment_geometry(q,ref,T)
    assert ok and ov==1.0 and rmse<1e-10

def test_wrong_translation_fails():
    rng=np.random.default_rng(14)
    q=rng.normal(size=(500,3));r=q.copy()
    T=np.eye(4);T[:3,3]=[50,0,0]
    ok,ov,_=verify_alignment_geometry(q,r,T)
    assert not ok and ov==0

def test_nonfinite_fails_closed():
    q=np.zeros((50,3));q[0,0]=np.nan
    assert verify_alignment_geometry(q,np.zeros((50,3)),np.eye(4))[0] is False

def test_raises_on_invalid_policy():
    with pytest.raises(ValueError): GeometryVerificationPolicy(min_overlap=1.5)
    with pytest.raises(ValueError): GeometryVerificationPolicy(max_points=0)

def test_deterministic_subsample():
    rng=np.random.default_rng(77);q=rng.random((10000,3))*50
    a=verify_alignment_geometry(q,q,np.eye(4))
    b=verify_alignment_geometry(q,q,np.eye(4))
    assert a==b and a[0]
