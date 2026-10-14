import numpy as np
import pytest
from scipy.spatial import cKDTree
from localization.surfel_6dof import Surfel6DConfig, align_surfel_6dof


def test_exact_parallel_correspondences():
    rng=np.random.default_rng(2026)
    dst=rng.normal(size=(12000,3))*20
    src=rng.normal(size=(4000,3))*20
    for leaf in (16,32,64):
        tree=cKDTree(dst,leafsize=leaf,balanced_tree=False,compact_nodes=False)
        d1,i1=tree.query(src,k=1,workers=1)
        d4,i4=tree.query(src,k=1,workers=4)
        np.testing.assert_array_equal(i1,i4)
        np.testing.assert_array_equal(d1,d4)


def test_solver_parallel_equivalence():
    rng=np.random.default_rng(17)
    pts=rng.normal(size=(1300,3))*6
    cov=np.tile([.001,0.,0.,.01,0.,.02],(len(pts),1))
    counts=np.full(len(pts),20)
    T=np.eye(4);T[0,3]=.08
    results=[]
    for workers,leaf in ((1,16),(2,16),(4,16)):
        cfg=Surfel6DConfig(iterations=5,max_points=1000,min_inliers=30,nn_workers=workers,tree_leafsize=leaf)
        results.append(align_surfel_6dof(pts,cov,counts,pts,T,cfg,fast_tree=True))
    for v in results[1:]:
        np.testing.assert_allclose(results[0].T_target_source,v.T_target_source,rtol=0,atol=1e-10)
        assert results[0].converged==v.converged
        assert results[0].failure_reason==v.failure_reason


def test_invalid_parallel_config():
    pts=np.zeros((100,3)); cov=np.zeros((100,6)); counts=np.ones(100)
    for kw in ({'nn_workers':0},{'nn_workers':-2},{'tree_leafsize':0}):
        with pytest.raises(ValueError):
            align_surfel_6dof(pts,cov,counts,pts,np.eye(4),Surfel6DConfig(**kw))
