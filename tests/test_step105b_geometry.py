from __future__ import annotations
import numpy as np
import pytest
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D,merge_occupied_voxels
from versioned_map.geometry_benchmark_105b import verify_geometry,run_geometry_benchmark


class Prepared:
    def __init__(self):
        self.spec=WorldSpec(32,32,16,.5)
        self.origin_local_m=(0.,0.,0.)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']
        cube=[(x,y,z) for x in range(4,8) for y in range(4,8) for z in range(2,6)]
        # A stable surface, a moving patch and a rare reappearing observation.
        stable=[(x,y,9) for x in range(12,18) for y in range(6,10)]
        self.session_codes=[]
        for i in range(5):
            pts=set(cube+stable+[(20+i,20,4),(1,1,1)] if i!=2 else cube+stable+[(22,20,4)])
            a=np.asarray(sorted(pts),dtype=np.uint64)
            self.session_codes.append(linear_encode(a[:,0],a[:,1],a[:,2],self.spec))


@pytest.fixture
def db(tmp_path):
    src=VersionedStore(tmp_path/'source');src.create(Prepared(),'official',8)
    merged=Geometry4D.build(src,tmp_path/'geometry',k=4)
    return src,merged


def test_spatial_sibling_merging_and_k_cap():
    dense=np.asarray([(x,y,z) for x in range(4) for y in range(4) for z in range(4)],dtype=np.int32)
    fine=merge_occupied_voxels(dense,side=8,k=1)
    coarse=merge_occupied_voxels(dense,side=8,k=4)
    assert len(fine)==64
    assert len(coarse)==16
    assert np.all(np.prod(coarse[:,3:],axis=1)<=4)
    assert np.sum(np.prod(coarse[:,3:],axis=1))==64
    assert np.any(coarse[:,3:]>1)
    # Each axis can merge when K allows; K=4 cannot merge 2x2x2=8 voxels.


def test_not_invent_occupancy_from_sparse():
    sparse=np.asarray([(1,1,1),(3,3,3),(6,4,2)],dtype=np.int32)
    result=merge_occupied_voxels(sparse,side=8,k=4)
    assert len(result)==3
    assert np.all(result[:,3:]==1)


def test_build_queries_and_both_encodings(db):
    src,merged=db
    stats=merged.storage()['stats']
    assert stats['spatial_merge_count']>0
    assert stats['distinct_spatial_blocks_over_time']<stats['spatial_blocks_across_sessions']
    assert stats['block_version_runs']>=stats['distinct_spatial_blocks_over_time']
    reopened=Geometry4D(merged.root)
    assert verify_geometry(src.root,reopened.root,queries=35,seed=75,full=True)['full_maps_verified']==5
    for session in src.all_versions():
        for bounds in ((0,8,0,8,0,8),(7,17,5,14,2,10),(0,64,0,64,0,32),(-20,3,-5,9,-10,5)):
            expected=src.query(session,bounds)
            for mode in ('bst','morton'):
                np.testing.assert_array_equal(reopened.query(session,bounds,mode=mode),expected)
    # Author's actual 4D cell IDs are returned for merged geometry, not just tile labels.
    ids=reopened.four_d_cells('2012-02-02',(0,32,0,32,0,32),limit=20)
    assert ids
    from versioned_map.author import author_module
    Cell=author_module('region_algebra').Cell
    assert all(Cell.from_id(int(i,16)).flags==15 for i in ids)


def test_reject_duplicates_and_bad_k():
    with pytest.raises(ValueError):merge_occupied_voxels(np.asarray([[0,0,0],[0,0,0]]),8,4)
    with pytest.raises(ValueError):merge_occupied_voxels(np.asarray([[0,0,0]]),8,0)


def test_checksum_and_fair_benchmark(db,tmp_path):
    src,merged=db
    k1=Geometry4D.build(src,tmp_path/'k1',k=1)
    result=run_geometry_benchmark(src.root,merged.root,tmp_path/'bench',queries=9,repeats=2,seed=16,unmerged=k1.root)
    assert result
    assert (tmp_path/'bench'/'storage.json').exists()
    assert (tmp_path/'bench'/'trials.csv').exists()
    blob=next((merged.root/'blobs').iterdir())
    blob.write_bytes(b'broken')
    with pytest.raises(IOError):Geometry4D(merged.root).query('2012-01-15',(0,64,0,64,0,32),mode='morton')


def test_incremental_append_same_as_full_build(tmp_path):
    src=VersionedStore(tmp_path/'source');src.create(Prepared(),'official',8)
    names=list(src.all_versions())
    target=Geometry4D.build(src,tmp_path/'incremental',k=4,through_session=names[0])
    for name in names[1:]:
        info=target.append_from_store(src,name)
        assert info['index_rebuilt']
        for seen in target.meta['sessions']:
            b=(0,64,0,64,0,32)
            np.testing.assert_array_equal(target.query(seen,b),src.query(seen,b))
    fresh=Geometry4D.build(src,tmp_path/'fresh',k=4)
    assert all(target.meta['stats'][key]==fresh.meta['stats'][key] for key in ('input_voxels_across_sessions','spatial_blocks_across_sessions','spatial_merge_count','tiles_with_geometry'))
    for name in names:
        b=(0,64,0,64,0,32)
        np.testing.assert_array_equal(target.query(name,b),fresh.query(name,b))
    assert target.compact_blobs()['reclaimed_bytes']>=0
    assert target.storage()['geometry_payload_bytes']>0
