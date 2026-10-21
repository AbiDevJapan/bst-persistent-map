"""Step 11.3 invariants: exact overlaps, sparse counters, copy-on-write history."""
from __future__ import annotations

import hashlib
import zlib

import numpy as np
import pytest

from versioned_map.correction import CorrectionMap, Pose
from versioned_map.correction.count_deltas import (
    apply_deltas, pack_counts, signed_deltas, unpack_counts,
)
from versioned_map.correction.voxel import COUNT_DTYPE, encode_indices, decode_indices

BOUNDS = (-128,128,-128,128,-128,128)


def rec(name, pts, pose=None):
    return {'source_id':name,'robot_id':'r', 'observed_at':100,
            'points':np.asarray(pts,dtype=np.float64).reshape(-1,3),
            'pose':Pose.identity() if pose is None else pose}


def reference_codec_hash(indices):
    mask=bytearray(4096)
    for ix in set(map(int,indices)):
        mask[ix//8] |= 1 << (ix%8)
    return hashlib.sha256(mask).digest()


@pytest.mark.parametrize('n',[0,1,60,5000,32768])
def test_codec_canonical_hash_unchanged(n):
    rng=np.random.default_rng(113)
    indices=rng.integers(0,32768,size=n,dtype=np.uint16)
    enc,payload,digest,count=encode_indices(indices)
    assert digest == reference_codec_hash(indices)
    assert count==len(np.unique(indices))
    assert np.array_equal(decode_indices(enc,payload),np.unique(indices))


def test_signed_deltas_cancel_across_sources():
    ids,delta=signed_deltas([np.array([0,1,2],dtype='<u2')],
                            [np.array([1,2,3],dtype='<u2')])
    assert ids.tolist()==[0,3]
    assert delta.tolist()==[-1,1]


def test_sparse_counter_legacy_binary_compatibility():
    before=np.array([(0,1),(10,2),(123,100)],dtype=COUNT_DTYPE)
    # Legacy format was a zlib stream of sorted (uint16,uint32) pairs.
    legacy=zlib.compress(before.tobytes(),level=6)
    assert np.array_equal(unpack_counts(legacy),before)
    idx,delta=signed_deltas([np.array([0,10],dtype='<u2')],
                            [np.array([5,10],dtype='<u2')])
    now=apply_deltas(before,idx,delta)
    assert now.tolist()==[(5,1),(10,2),(123,100)]
    assert np.array_equal(unpack_counts(pack_counts(now)),now)
    with pytest.raises(RuntimeError):
        apply_deltas(before,np.array([8],dtype='<u2'),np.array([-1],dtype=np.int64))
    with pytest.raises(OverflowError):
        over=np.array([(1,2**32-1)],dtype=COUNT_DTYPE)
        apply_deltas(over,np.array([1],dtype='<u2'),np.array([1],dtype=np.int64))


def test_pose_only_revision_no_source_tile_or_new_page(tmp_path):
    with CorrectionMap(tmp_path/'map.sqlite') as db:
        r1=db.add_sources([rec('a',[[0.1,0.1,0.1]])]).revision
        baseline=db.stats()
        tx=db.con.total_changes
        r2=db.correct_poses({'a':Pose.translation(0.05,0,0)})
        assert r2.revision==r1+1
        assert r2.tiles_modified==0 and r2.tiles_touched==0
        assert db.stats()['historical_tile_records']==baseline['historical_tile_records']
        assert db.stats()['unique_compressed_pages']==baseline['unique_compressed_pages']
        assert db.query(BOUNDS,r1).tolist()==db.query(BOUNDS,r2.revision).tolist()
        pose=db._source_pose_asof('a',r2.revision)
        assert pose.matrix[0,3]==pytest.approx(.05)
        assert db.verify(r1) and db.verify(r2.revision)


def test_overlap_occupancy_unchanged_but_counters_updated(tmp_path):
    with CorrectionMap(tmp_path/'db.sqlite') as db:
        initial=db.add_sources([rec('A',[[0,0,0]]),rec('B',[[0,0,0]])]).revision
        baseline=db.stats()
        update=db.correct_poses({'A':Pose.translation(0.5,0,0)})
        # There are still contributors to voxel 0, but source A moved to voxel 1.
        assert db.query(BOUNDS).tolist()==[[0,0,0],[1,0,0]]
        assert db.query(BOUNDS,initial).tolist()==[[0,0,0]]
        # Moving B to voxel 1 must now empty voxel 0: checks changed counters.
        db.correct_poses({'B':Pose.translation(.5,0,0)})
        assert db.query(BOUNDS).tolist()==[[1,0,0]]
        assert db.verify(initial) and db.verify()


def test_count_only_update_without_new_geometry_page(tmp_path):
    with CorrectionMap(tmp_path/'db.sqlite') as db:
        db.add_sources([rec('A',[[0,0,0]]),rec('B',[[0,0,0]]),rec('C',[[.5,0,0]])])
        # Move A from voxel 0 to voxel 1, while moving C from 1 to 0.
        # Occupancy stays {0,1}, but the count table changes / can be used later.
        stats=db.stats();first=db.revision
        out=db.correct_poses({'A':Pose.translation(.5,0,0),'C':Pose.translation(-.5,0,0)})
        assert out.tiles_modified==0
        assert out.tiles_touched==0  # net deltas cancel in each tile
        assert db.stats()['historical_tile_records']==stats['historical_tile_records']
        assert db.verify(first) and db.verify(out.revision)
        db.correct_poses({'B':Pose.translation(1,0,0)})
        assert db.query(BOUNDS).tolist()==[[0,0,0],[1,0,0],[2,0,0]]
        assert db.verify()


def test_subvoxel_correction_and_history_across_negative_tiles(tmp_path):
    with CorrectionMap(tmp_path/'db.sqlite') as db:
        r1=db.add_sources([rec('A',[[-16.1,0,0],[-.1,0,0],[0,0,0],[16,0,0]]),
                           rec('B',[[-16.1,0,0],[-.1,0,0],[0,0,0],[16,0,0]])]).revision
        r2=db.correct_poses({'A':Pose.translation(2,0,0)}).revision
        r3=db.correct_poses({'A':Pose.translation(0,0,0)}).revision
        assert db.verify(r1) and db.verify(r2) and db.verify(r3)
        assert np.array_equal(db.query(BOUNDS,r1),db.query(BOUNDS,r3))
        assert db.diff(BOUNDS,r1,r3)['symmetric_difference']==0


def test_randomized_batch_matches_oracle_and_legacy_counts(tmp_path):
    rng=np.random.default_rng(113113)
    with CorrectionMap(tmp_path/'db.sqlite') as db:
        records=[]
        for i in range(20):
            p=rng.normal(size=(30,3))*2
            records.append(rec(f's{i}',p))
        db.add_sources(records)
        for j in range(8):
            ids=rng.choice(20,5,replace=False)
            pose={f's{i}':Pose.translation(*rng.uniform(-2,2,size=3)) for i in ids}
            out=db.correct_poses(pose)
            assert db.verify(out.revision)
        for rev in (0,1,2,5,9):
            assert db.verify(rev)
