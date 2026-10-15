import numpy as np

from types import SimpleNamespace
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from representations.packed_scale import merge_coords_fast
from representations.step83.time_local_core import TimeLocalAdaptiveCore
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from localization.step911 import GeometryGateProfile
from localization.step912 import (
    build_native_surfel_hierarchy,
    build_hierarchical_lod_view,
    HierarchicalLODBSTIndex,
)



def _small_fixture():
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
    return prepared,session_blocks,core,bst,aligned

def _hierarchy():
    prepared,session_blocks,core,bst,aligned=_small_fixture()
    h=build_native_surfel_hierarchy(
        aligned,prepared,session_blocks,core,bst,0,
        normal_min_points=1,normal_max_curvature=1.0,
        normal_min_horizontal_component=0.0,progress=False,
    )
    return prepared,session_blocks,core,bst,aligned,h


def test_hierarchy_reproduces_native_root_payload_and_coverage():
    _,_,_,_,_,h=_hierarchy()
    assert h.leaf_count==4
    assert h.internal_count==2
    assert len(h.root_node_ids)==len(h.block_map.blocks)==2
    assert np.allclose(h.node_centroids(h.root_node_ids),h.block_map.centroids)
    assert int(h.node_volume(h.root_node_ids).sum())==h.leaf_count


def test_k1_is_exact_native_root_view():
    *_,h=_hierarchy()
    p=GeometryGateProfile('balanced',0.05,0.025,0.95,2.0)
    v=build_hierarchical_lod_view(h,p,1)
    assert np.array_equal(v.selected_node_ids,h.root_node_ids)
    assert np.all(v.root_counts==1)
    assert v.map_elements==len(h.root_node_ids)


def test_k2_splits_only_bad_parent_and_preserves_exact_coverage():
    _,_,_,bst,_,h=_hierarchy()
    # The fixture has one coherent 2-voxel parent and one parent whose leaf
    # normals are orthogonal. Loose plane/extent thresholds isolate coherence.
    p=GeometryGateProfile('coherence',1.0,1.0,0.90,4.0)
    v=build_hierarchical_lod_view(h,p,2)
    assert v.root_counts.tolist()==[1,2]
    assert v.map_elements==3
    assert int(h.node_volume(v.selected_node_ids).sum())==h.leaf_count
    assert int(v.root_counts.max())<=2
    idx=HierarchicalLODBSTIndex(v,bst,0)
    got=idx.query_radius_ids(.75,.75,5.0)
    assert len(got)==3
    assert idx.last_stats['active_blocks']==2


def test_k_cap_power_of_two_and_no_forced_refinement():
    *_,h=_hierarchy()
    good=GeometryGateProfile('allgood',10.0,10.0,0.0,100.0)
    v=build_hierarchical_lod_view(h,good,8)
    # K is an upper bound: good roots remain one primitive each.
    assert np.all(v.root_counts==1)
    try:
        build_hierarchical_lod_view(h,good,3)
    except ValueError:
        pass
    else:
        raise AssertionError('non-power-of-two K must be rejected')
