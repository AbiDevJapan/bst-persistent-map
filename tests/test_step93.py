from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from representations.packed_scale import merge_coords_fast
from representations.step83.time_local_core import TimeLocalAdaptiveCore
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from localization.step911 import GeometryGateProfile
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.step913 import materialize_lod_view
from localization.step93 import build_causal_persistence, PersistenceWeightedLODBSTIndex
from benchmark.step93_benchmark import (
    _paired_history_outcomes, _verify_null_equivalence, _persistence_gates,
    UNWEIGHTED, PERSISTENT,
)
from benchmark.step92_benchmark import _session_summary


def _fixture(session_codes_override=None):
    spec=WorldSpec(2.0,2.0,1.0,0.5)
    coords=np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0]],dtype=np.int32)
    codes=np.sort(linear_encode(coords[:,0],coords[:,1],coords[:,2],spec))
    x=(codes//(spec.ny*spec.nz)).astype(np.int32); r=codes%(spec.ny*spec.nz)
    y=(r//spec.nz).astype(np.int32); z=(r%spec.nz).astype(np.int32); coords=np.column_stack((x,y,z))
    s0=codes.copy(); s1=codes[:3].copy(); s2=codes[[0,2]].copy()
    sess=session_codes_override or [s0,s1,s2]
    prepared=SimpleNamespace(spec=spec,session_codes=sess,session_names=['a','b','c'],origin_local_m=(0.,0.,0.))
    session_blocks=[merge_coords_fast(np.column_stack(np.unravel_index(c,(spec.nx,spec.ny,spec.nz))).astype(np.int32),(spec.nx,spec.ny,spec.nz)) for c in sess]
    core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=False); bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())
    counts=np.full(4,20,dtype=np.uint32); cent=(coords.astype(np.float32)+.5)*.5
    cov=np.tile(np.array([[.001,0,0,1.0,0,1.0]],dtype=np.float32),(4,1)); normals=np.tile(np.array([[1,0,0]],dtype=np.float32),(4,1))
    aligned=SimpleNamespace(spec=spec,origin_local_m=(0.,0.,0.),linear_codes=codes,counts=counts,centroids=cent,cov6=cov,normals=normals,curvature=np.full(4,.001,dtype=np.float32),normal_valid=np.ones(4,dtype=np.uint8))
    h=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,0,normal_min_points=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0,progress=False)
    view=build_hierarchical_lod_view(h,GeometryGateProfile('all',1.0,1.0,0.0,4.0),2)
    return prepared,bst,h,view


def test_causal_first_query_is_exact_all_one_control():
    prepared,_,_,view=_fixture(); c=build_causal_persistence(view,prepared,0,[1,2])
    assert np.all(c.scores_q8[1]==255)
    assert c.stats[0]['prior_session_count']==1
    assert c.score_bytes_per_active_session()==len(view.selected_node_ids)


def test_causal_scores_exclude_query_and_future_sessions():
    prepared,_,_,view=_fixture(); a=build_causal_persistence(view,prepared,0,[2]).scores_q8[2]
    # Change session 2 (the query session) completely; Jan->session2 may use only sessions 0 and 1.
    changed=SimpleNamespace(**prepared.__dict__)
    changed.session_codes=[prepared.session_codes[0],prepared.session_codes[1],np.empty(0,dtype=np.uint64)]
    b=build_causal_persistence(view,changed,0,[2]).scores_q8[2]
    assert np.array_equal(a,b)
    assert np.any(a<255)


def test_persistence_index_preserves_retrieved_geometry_ids():
    prepared,bst,_,view=_fixture(); compact=materialize_lod_view(view); causal=build_causal_persistence(view,prepared,0,[2])
    pidx=PersistenceWeightedLODBSTIndex(compact,bst,0,causal.scores_q8[2],2)
    base=pidx.base
    for q in [(.5,.5,5.0),(1.25,.25,.8)]:
        a=base.query_radius_ids(*q); b=pidx.query_radius_ids(*q)
        assert np.array_equal(a,b)
        assert pidx.last_stats['returned']==len(b)


def _row(geom,session,success,total=280.0,scan=0):
    return {'map_geometry':geom,'scenario':'cross_session','query_session':session,'perturbation':'xy0.5_yaw1','success':success,'converged':True,'original_scan_index':scan,'final_translation_error_m':.2 if success else .7,'final_yaw_error_deg':.7 if success else 1.2,'retrieval_ms':10.,'map_prep_ms':240.,'match_ms':30.,'total_ms':total,'query_points':3000,'local_map_primitives':50000,'correspondences':2800,'normal_correspondences':1200,'final_plane_rmse_m':.12,'mean_abs_plane_residual_m':.09,'iterations':15,'estimated_x':1.,'estimated_y':2.,'estimated_yaw':.1,'failure_reason':''}


def test_history_paired_outcomes_and_gate_use_later_sessions():
    rows=[]
    # First session exact null; later sessions have one net persistence win.
    pats={'b':([1,0],[1,0]),'c':([1,0],[1,1]),'d':([1,1],[1,1])}
    for s,(u,p) in pats.items():
        for i,(x,y) in enumerate(zip(u,p)):
            rows.append(_row(UNWEIGHTED,s,bool(x),280,scan=i)); rows.append(_row(PERSISTENT,s,bool(y),285,scan=i))
    paired=_paired_history_outcomes(rows); later=next(r for r in paired if r['query_session']=='LATER')
    assert later['net_persistence_success_difference']==1
    ss=_session_summary(rows); gate=_persistence_gates(ss,paired,'b')[0]
    assert gate['strict_success_gain_observed']
    assert gate['primitive_count_preserved']
    assert gate['meets_runtime_overhead_1.10']


def test_null_equivalence_verifier_detects_exact_control():
    rows=[]
    for i in range(3):
        a=_row(UNWEIGHTED,'b',True,scan=i); b=dict(a); b['map_geometry']=PERSISTENT
        rows.extend([a,b])
    v=_verify_null_equivalence(rows,'b')
    assert v['passed'] and v['max_abs_float_difference']==0.0

def test_all_one_persistence_matcher_is_exact_frozen_null_path():
    from scipy.spatial import cKDTree
    from localization.step90 import Pose3DOF
    from localization.step901 import PointToPlaneICP3DOFConfig
    from localization.step902 import PyramidLevelConfig
    from localization.step903b import PreparedSurfelLevel, scan_to_map_coarse_to_fine_surfel_3dof
    from localization.step93 import PreparedPersistentSurfelLevel, scan_to_map_coarse_to_fine_persistent_3dof
    # Vertical x=0 plane gives horizontal normals useful for x/y/yaw matching.
    mp=np.array([[0.,y,z] for y in (-1.,0.,1.) for z in (0.,1.,2.)],dtype=float)
    normals=np.tile(np.array([[1.,0.,0.]]),(len(mp),1)); valid=np.ones(len(mp),dtype=bool); cnt=np.ones(len(mp),dtype=np.uint32)
    lev=PyramidLevelConfig('only',scan_voxel_m=.01,map_voxel_m=.01,max_correspondence_m=2.,max_iterations=2,trim_fraction=1.0,normal_max_radius_m=2.,max_step_translation_m=1.,max_step_yaw_deg=5.)
    tree=cKDTree(mp); base=[PreparedSurfelLevel(lev,mp,normals,valid,cnt,tree)]; pers=[PreparedPersistentSurfelLevel(lev,mp,normals,valid,cnt,tree,np.ones(len(mp),dtype=np.float32))]
    cfg=PointToPlaneICP3DOFConfig(min_correspondences=3,normal_min_neighbors=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0)
    pts=mp.copy(); init=Pose3DOF(.1,0.,0.)
    a=scan_to_map_coarse_to_fine_surfel_3dof(pts,base,init,0.,0.,0.,cfg)
    b=scan_to_map_coarse_to_fine_persistent_3dof(pts,pers,init,0.,0.,0.,cfg)
    assert a.pose==b.pose
    assert a.iterations==b.iterations and a.correspondences==b.correspondences
    assert a.rmse_m==b.rmse_m and a.mean_abs_plane_residual_m==b.mean_abs_plane_residual_m
