import numpy as np
import pytest
from types import SimpleNamespace

from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from representations.packed_scale import merge_coords_fast
from representations.step83.time_local_core import TimeLocalAdaptiveCore
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step911 import GeometryGateProfile
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view, HierarchicalLODBSTIndex
from localization.step913 import materialize_lod_view, MaterializedLODBSTIndex, verify_materialization


def _fixture():
    spec=WorldSpec(2.0,2.0,1.0,0.5)
    coords=np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0]],dtype=np.int32)
    codes=np.sort(linear_encode(coords[:,0],coords[:,1],coords[:,2],spec))
    x=(codes//(spec.ny*spec.nz)).astype(np.int32)
    r=codes%(spec.ny*spec.nz); y=(r//spec.nz).astype(np.int32); z=(r%spec.nz).astype(np.int32)
    coords=np.column_stack((x,y,z))
    prepared=SimpleNamespace(spec=spec,session_codes=[codes],session_names=['a'],origin_local_m=(0.,0.,0.))
    session_blocks=[merge_coords_fast(coords,(spec.nx,spec.ny,spec.nz))]
    core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=False)
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())
    counts=np.full(4,20,dtype=np.uint32)
    cent=(coords.astype(np.float32)+.5)*.5
    cov=np.tile(np.array([[.001,0,0,1.0,0,1.0]],dtype=np.float32),(4,1))
    normals=np.array([[1,0,0],[1,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    aligned=SimpleNamespace(spec=spec,origin_local_m=(0.,0.,0.),linear_codes=codes,counts=counts,centroids=cent,cov6=cov,normals=normals,curvature=np.full(4,.001,dtype=np.float32),normal_valid=np.ones(4,dtype=np.uint8))
    h=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,0,normal_min_points=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0,progress=False)
    p=GeometryGateProfile('coherence',1.0,1.0,0.90,4.0)
    v=build_hierarchical_lod_view(h,p,2)
    return bst,h,v


def test_materialized_payload_is_exact_and_hierarchy_free():
    _,h,v=_fixture()
    m=materialize_lod_view(v)
    ids=v.selected_node_ids
    assert np.array_equal(m.counts,h.node_counts(ids).astype(np.uint32))
    assert np.array_equal(m.centroids,h.node_centroids(ids).astype(np.float32))
    assert np.array_equal(m.cov6,h.node_cov6(ids).astype(np.float32))
    assert np.array_equal(m.root_starts,v.root_starts)
    assert np.array_equal(m.root_counts,v.root_counts)
    assert np.array_equal(m.global_block_ids,h.block_map.global_block_ids)
    assert not hasattr(m,'hierarchy')
    assert not hasattr(m,'selected_node_ids')
    assert m.runtime_nbytes()==m.geometry_payload_nbytes()+m.directory_nbytes()


def test_materialized_query_matches_hierarchy_query_exactly():
    bst,h,v=_fixture(); m=materialize_lod_view(v)
    old=HierarchicalLODBSTIndex(v,bst,0); new=MaterializedLODBSTIndex(m,bst,0)
    src=np.asarray(v.selected_node_ids,dtype=np.int64); order=np.argsort(src); ss=src[order]
    for cx,cy,r in [(.75,.75,5.0),(.25,.25,.55),(1.25,.25,.6)]:
        ids=old.query_radius_ids(cx,cy,r)
        pos=np.searchsorted(ss,ids)
        rows=order[pos] if len(ids) else np.empty(0,dtype=np.int64)
        got=new.query_radius_ids(cx,cy,r)
        assert np.array_equal(rows,got)
        assert old.last_stats['active_blocks']==new.last_stats['active_blocks']


def test_materialized_pyramid_is_numerically_identical():
    bst,h,v=_fixture(); m=materialize_lod_view(v)
    cfg=PointToPlaneICP3DOFConfig(min_correspondences=1,normal_min_neighbors=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0)
    ver=verify_materialization(v,m,bst,0,[(.75,.75),(1.25,.25)],5.0,DEFAULT_PYRAMID,cfg,max_pyramid_checks=2)
    assert ver.passed
    assert ver.checks==2


def test_count_width_is_compacted_without_value_loss():
    _,h,v=_fixture(); m=materialize_lod_view(v)
    assert m.counts.dtype==np.uint32
    assert np.array_equal(m.counts.astype(np.uint64),h.node_counts(v.selected_node_ids))


def test_materialization_verification_rejects_nonfinite_center_cleanly():
    bst,h,v=_fixture(); m=materialize_lod_view(v)
    cfg=PointToPlaneICP3DOFConfig(min_correspondences=1,normal_min_neighbors=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0)
    with pytest.raises(ValueError, match='non-finite materialization verification center'):
        verify_materialization(v,m,bst,0,[(float('nan'),.75)],5.0,DEFAULT_PYRAMID,cfg,max_pyramid_checks=1)
