from types import SimpleNamespace
import numpy as np

from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from representations.packed_scale import merge_coords_fast
from representations.step83.time_local_core import TimeLocalAdaptiveCore
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from localization.step911 import (
    GeometryGateProfile,
    parse_geometry_profiles,
    build_native_block_surfel_map_with_leaf_owners,
    build_native_geometry_descriptors,
    build_geometry_aware_virtual_view,
    GeometryAwareBSTVirtualIndex,
)


def _small_fixture():
    spec=WorldSpec(2.0,2.0,1.0,0.5)  # 4x4x2
    coords=np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0]],dtype=np.int32)
    codes=np.sort(linear_encode(coords[:,0],coords[:,1],coords[:,2],spec))
    # Reorder coords to code order so payload rows align exactly with codes.
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
    aligned=SimpleNamespace(
        spec=spec,origin_local_m=(0.,0.,0.),linear_codes=codes,counts=counts,
        centroids=cent,cov6=cov,normals=normals,curvature=np.full(4,.001,dtype=np.float32),
        normal_valid=np.ones(4,dtype=np.uint8),
    )
    return prepared,session_blocks,core,bst,aligned


def test_parse_geometry_profiles():
    p=parse_geometry_profiles('a:0.05:0.02:0.95:2;b:0.1:0.05:0.8:4')
    assert [x.name for x in p]==['a','b']
    assert p[0].max_plane_sigma_m==0.05


def test_leaf_owner_mapping_reproduces_native_blocks():
    prepared,session_blocks,core,bst,aligned=_small_fixture()
    bm,owners=build_native_block_surfel_map_with_leaf_owners(
        aligned,prepared,session_blocks,core,bst,0,
        normal_min_points=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0,progress=False,
    )
    assert len(bm.blocks)==2
    assert len(owners)==4
    assert set(owners.tolist())==set(bm.global_block_ids.tolist())
    vals,counts=np.unique(owners,return_counts=True)
    assert counts.tolist()==[2,2]


def test_geometry_gate_keeps_coherent_parent_and_falls_back_incoherent_parent():
    prepared,session_blocks,core,bst,aligned=_small_fixture()
    bm,owners=build_native_block_surfel_map_with_leaf_owners(
        aligned,prepared,session_blocks,core,bst,0,
        normal_min_points=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0,progress=False,
    )
    desc=build_native_geometry_descriptors(aligned,bm,owners,batch_size=10)
    prof=GeometryGateProfile('test',1.0,1.0,0.90,4.0)
    view,_=build_geometry_aware_virtual_view(aligned,bm,desc,prof)
    # One two-voxel parent has coherent normals -> one primitive; the other
    # has orthogonal base normals -> falls back to its two voxel surfels.
    assert view.kept_native_blocks==1
    assert view.fallback_parent_blocks==1
    assert view.fallback_voxels==2
    assert view.map_elements==3

    idx=GeometryAwareBSTVirtualIndex(view,bst,0)
    got=idx.query_radius_ids(.75,.75,5.0)
    assert len(got)==3
    assert idx.last_stats['active_blocks']==2
    assert idx.last_stats['fallback_parent_blocks']==1


def test_single_voxel_native_parent_is_never_refined():
    prepared,session_blocks,core,bst,aligned=_small_fixture()
    bm,owners=build_native_block_surfel_map_with_leaf_owners(
        aligned,prepared,session_blocks,core,bst,0,
        normal_min_points=1,normal_max_curvature=1.0,normal_min_horizontal_component=0.0,progress=False,
    )
    desc=build_native_geometry_descriptors(aligned,bm,owners,batch_size=10)
    # Synthetic direct descriptor check for the invariant.
    desc.voxel_volume[0]=1
    desc.plane_sigma_m[0]=999
    desc.relative_planarity[0]=999
    prof=GeometryGateProfile('strict',1e-6,1e-6,1.0,.1)
    view,_=build_geometry_aware_virtual_view(aligned,bm,desc,prof)
    assert view.keep_native[0]
