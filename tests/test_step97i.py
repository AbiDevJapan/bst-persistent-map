import numpy as np
from types import SimpleNamespace
from localization.compact_map import IndexedMaterializedLODBSTIndex
from localization.surfel_6dof import align_surfel_6dof, Surfel6DConfig


def test_index_returns_identical_ordered_rows():
    rng=np.random.default_rng(11)
    points=rng.uniform(-30,30,(12000,3))
    obj=SimpleNamespace(centroids=points,directory_nbytes=lambda: 0)
    fake_bst=SimpleNamespace(core=None)
    fast=IndexedMaterializedLODBSTIndex(obj,fake_bst,0)
    assert fast.extra_index_nbytes_estimate > 0
    for cx,cy,r in [(0,0,4),(20,-12,23),(100,100,2),(0,0,0),(1.5,-2.5,50)]:
        expected=np.flatnonzero((points[:,0]-cx)**2+(points[:,1]-cy)**2 <= r*r)
        got=fast.query_radius_ids(cx,cy,r)
        np.testing.assert_array_equal(expected,got)


def test_index_exact_radius_boundary():
    pts=np.array([[3.,4.,0.],[3.,4.001,0.],[0,0,0],[-3,-4,9]],dtype=float)
    idx=IndexedMaterializedLODBSTIndex(SimpleNamespace(centroids=pts,directory_nbytes=lambda:0),SimpleNamespace(core=None),0)
    np.testing.assert_array_equal(idx.query_radius_ids(0,0,5),[0,2,3])


def test_fast_tree_same_registration_result():
    rng=np.random.default_rng(4)
    pts=rng.normal(size=(500,3))*5
    # Distinct positions, realistic valid enough covariance
    cov=np.tile([.001,0,0,.01,0,.02],(len(pts),1))
    counts=np.full(len(pts),20)
    initial=np.eye(4)
    cfg=Surfel6DConfig(iterations=4,max_points=250,min_inliers=25)
    a=align_surfel_6dof(pts,cov,counts,pts.copy(),initial,cfg,fast_tree=False)
    b=align_surfel_6dof(pts,cov,counts,pts.copy(),initial,cfg,fast_tree=True)
    np.testing.assert_allclose(a.T_target_source,b.T_target_source,rtol=0,atol=1e-9)
    assert a.converged == b.converged
    assert a.failure_reason == b.failure_reason
