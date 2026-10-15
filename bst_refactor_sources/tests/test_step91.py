from types import SimpleNamespace
import numpy as np

from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from representations.packed_scale import merge_coords_fast
from representations.step83.time_local_core import TimeLocalAdaptiveCore
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from localization.step91 import merge_voxel_surfels_to_blocks, finalize_normals, NativeBlockSurfelMap, NativeBlockSurfelIndex


def test_payload_merge_matches_exact_spatial_tiling_and_statistics():
    coords=np.array([[0,0,0],[1,0,0]],dtype=np.int32)
    counts=np.array([10,30],dtype=np.uint32)
    cent=np.array([[0.2,0.0,0.0],[1.2,0.0,0.0]],dtype=np.float32)
    cov=np.zeros((2,6),dtype=np.float32)
    b,c,m,cv=merge_voxel_surfels_to_blocks(coords,counts,cent,cov,(4,2,2))
    ref=merge_coords_fast(coords,(4,2,2))
    assert np.array_equal(np.sort(b,axis=0),np.sort(ref,axis=0)) or {tuple(x) for x in b}=={tuple(x) for x in ref}
    assert len(b)==1
    assert int(c[0])==40
    assert np.allclose(m[0],[0.95,0.0,0.0],atol=1e-6)
    # Population variance from two point masses with weights 10/30.
    assert abs(float(cv[0,0])-0.1875)<1e-5


def test_finalize_normals_accepts_vertical_plane_and_rejects_ground_plane():
    # xx small => normal x, useful for planar x/y localization.
    cov=np.array([[0.001,0,0,1.0,0,1.0],[1.0,0,0,1.0,0,0.001]],dtype=np.float32)
    n,curv,valid=finalize_normals(cov,np.array([20,20]),min_points=6,max_curvature=.2,min_horizontal_component=.2)
    assert valid.tolist()==[1,0]
    assert abs(float(n[0,0]))>.9


def test_native_block_index_uses_time_index_and_returns_block_rows():
    spec=WorldSpec(2.0,1.0,1.0,0.5)  # 4x2x2
    c0=linear_encode(np.array([0,1]),np.array([0,0]),np.array([0,0]),spec)
    c1=linear_encode(np.array([0]),np.array([0]),np.array([0]),spec)
    prepared=SimpleNamespace(spec=spec,session_codes=[np.sort(c0),np.sort(c1)],session_names=['a','b'])
    session_blocks=[merge_coords_fast(np.array([[0,0,0],[1,0,0]],dtype=np.int32),(spec.nx,spec.ny,spec.nz)),merge_coords_fast(np.array([[0,0,0]],dtype=np.int32),(spec.nx,spec.ny,spec.nz))]
    core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=False)
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())
    # Find session-0 active global IDs and attach a trivial block surfel.
    ids,_,_,_=core._select_mode((0,3,0,1,0,1),0,index_mode='time')
    ids=np.sort(ids.astype(np.int32))
    B=core.blocks[ids,:6]
    cent=np.column_stack(((B[:,0]+B[:,3]/2)*.5,(B[:,1]+B[:,4]/2)*.5,(B[:,2]+B[:,5]/2)*.5)).astype(np.float32)
    bm=NativeBlockSurfelMap(spec,(0.,0.,0.),ids,B,np.ones(len(ids),dtype=np.uint64),cent,np.zeros((len(ids),6),dtype=np.float32),np.tile(np.array([[1.,0.,0.]],dtype=np.float32),(len(ids),1)),np.zeros(len(ids),dtype=np.float32),np.ones(len(ids),dtype=np.uint8),bst,0,0.)
    idx=NativeBlockSurfelIndex(bm)
    got=idx.query_radius_ids(.5,.25,2.0)
    assert len(got)==len(ids)
    assert idx.last_stats['blocks_scanned']==len(ids)
