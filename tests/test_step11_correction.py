"""Step 11 reference invariants and randomized bitemporal correction tests."""
from __future__ import annotations
import sqlite3
import numpy as np
import pytest
from versioned_map.correction import CorrectionMap,Pose
from versioned_map.correction.voxel import encode_indices, decode_indices, full_mask, GRID_SIDE, tile_keyed_indices

ROI=(-100,100,-100,100,-20,20)


def make_db(tmp_path):
    return CorrectionMap(tmp_path/'map.sqlite',voxel_size=.5,origin=(0,0,0))


def source(sid,pts,pose=None,robot='robot-A',at=100):
    return dict(source_id=sid,robot_id=robot,observed_at=at,
                points=np.asarray(pts,dtype=np.float64).reshape(-1,3),
                pose=Pose.identity() if pose is None else pose)


def test_voxel_codec_exact_and_adaptive():
    for count in (0,1,4,31,512,8000,32768):
        ii=np.linspace(0,32767,count,dtype=int).astype('<u2') if count else np.empty(0,dtype='<u2')
        enc,data,hashval,n=encode_indices(ii)
        out=decode_indices(enc,data)
        assert n==len(np.unique(ii))
        assert np.array_equal(out,np.unique(ii))
        assert len(full_mask(enc,data))==4096
        assert len(hashval)==32


def test_negative_signed_grid_and_half_open(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('a',[[-.01,-.51,0],[0,0,0],[15.999,0,0],[16,0,0]])])
        vox=db.query(ROI)
        assert (-1,-2,0) in list(map(tuple,vox))
        assert (0,0,0) in list(map(tuple,vox))
        assert np.array_equal(db.query((-1,0,-2,-1,0,1)),[[-1,-2,0]])
        assert len(db.query((1,2,-2,-1,0,1)))==0
        assert db.verify()


def test_source_dedup_within_one_scan(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('a',[[.1,.1,.1],[.2,.2,.2],[.1,.1,.1]])])
        assert len(db.query(ROI))==1
        assert db.verify()


def test_overlap_correction_preserves_other_robot(tmp_path):
    with make_db(tmp_path) as db:
        initial=db.add_sources([source('a',[[0,0,0],[.5,0,0]]),
                                source('b',[[0,0,0]],robot='robot-B')])
        assert len(db.query(ROI))==2
        corrected=db.correct_poses({'a':Pose.translation(1,0,0)})
        assert db.query(ROI).tolist()==[[0,0,0],[2,0,0],[3,0,0]]
        assert db.query(ROI,initial.revision).tolist()==[[0,0,0],[1,0,0]]
        assert db.diff(ROI,initial.revision,corrected.revision)=={
            'added_observations':2,'no_longer_observed':1,'symmetric_difference':3,
            'left_revision':1,'right_revision':2}
        assert db.verify(initial.revision) and db.verify(corrected.revision)


def test_pose_revision_bitemporal(tmp_path):
    with make_db(tmp_path) as db:
        r1=db.add_sources([source('A',[[0,0,0]],at=100)])
        r2=db.add_sources([source('B',[[1,0,0]],at=200)])
        r3=db.correct_poses({'A':Pose.translation(1,0,0)})
        assert db.query_observed_history(ROI,observed_until=100,as_of_revision=r1.revision).tolist()==[[0,0,0]]
        assert db.query_observed_history(ROI,observed_until=100,as_of_revision=r3.revision).tolist()==[[2,0,0]]
        assert db.query_observed_history(ROI,observed_until=200,as_of_revision=r3.revision).tolist()==[[2,0,0]]
        assert len(db.query(ROI,r2.revision))==2
        assert db.verify(r2.revision) and db.verify(r3.revision)


def test_batched_corrections_do_not_lose_shared_cells(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[0,0,0]]),source('B',[[0,0,0]]),source('C',[[0,0,0]])])
        out=db.correct_poses({'A':Pose.translation(.5,0,0),'B':Pose.translation(-.5,0,0)})
        assert out.tiles_touched >= 1
        assert db.query(ROI).tolist()==[[-1,0,0],[0,0,0],[1,0,0]]
        assert db.verify()


def test_empty_tiles_and_historical_tombstones(tmp_path):
    with make_db(tmp_path) as db:
        r1=db.add_sources([source('A',[[0,0,0]])])
        r2=db.correct_poses({'A':Pose.translation(40,0,0)})
        assert db.query((-1,5,-1,5,-1,5),r2.revision).size==0
        assert db.query((-1,5,-1,5,-1,5),r1.revision).tolist()==[[0,0,0]]
        assert db.verify(r1.revision) and db.verify(r2.revision)


def test_idempotent_correction_no_new_geometry_page(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[0,0,0]])]);before=db.stats()
        new=db.correct_poses({'A':Pose.identity()})
        assert new.tiles_modified==0
        assert db.stats()['historical_tile_records']==before['historical_tile_records']
        assert db.verify()


def test_noncanonical_pose_rollback(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[0,0,0]])])
        pose=np.eye(4);pose[0,0]=2
        with pytest.raises(ValueError):db.correct_poses({'A':pose})
        assert db.revision==1
        assert db.verify()


def test_failed_batch_atomic_and_duplicate_source(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[0,0,0]])])
        with pytest.raises(sqlite3.IntegrityError):
            db.add_sources([source('B',[[1,0,0]]),source('A',[[2,0,0]])])
        assert db.revision==1
        assert db.stats()['sources']==1
        assert db.verify()


def test_reopen_preserves_history(tmp_path):
    path=tmp_path/'persistent.sqlite'
    with CorrectionMap(path) as db:
        db.add_sources([source('A',[[0,0,0]])])
        db.correct_poses({'A':Pose.translation(.5,0,0)})
    with CorrectionMap(path) as db:
        assert db.query(ROI,revision=1).tolist()==[[0,0,0]]
        assert db.query(ROI,revision=2).tolist()==[[1,0,0]]
        assert db.verify(1) and db.verify(2)


def test_arbitrary_rigid_rotation(tmp_path):
    p=np.eye(4);p[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[1.2,.2,0],[.2,1.2,0]])])
        db.correct_poses({'A':Pose(p)})
        assert db.verify()


def test_randomized_multiple_revisions_against_full_rebuild(tmp_path):
    rng=np.random.default_rng(11000)
    with make_db(tmp_path) as db:
        for n in range(14):
            center=rng.uniform(-10,10,size=3)
            pts=rng.normal(0,.5,size=(80,3))+center
            db.add_sources([source(f'S{n}',pts,robot=f'R{n%3}',at=100+n)])
        for k in range(12):
            ids=rng.choice(14,size=3,replace=False)
            corrections={f'S{i}':Pose.translation(*rng.uniform(-4,4,size=3)) for i in ids}
            result=db.correct_poses(corrections)
            assert db.verify(result.revision)
        for r in (0,1,3,7,11,14,17,21,26):
            assert db.verify(r)
            if r:
                assert np.array_equal(db.query(ROI,r),db.full_rebuild(ROI,r))


def test_empty_source_points_and_revisions(tmp_path):
    with make_db(tmp_path) as db:
        r=db.add_sources([source('empty',np.empty((0,3)))])
        assert r.tiles_touched==0
        assert db.query(ROI).size==0
        assert db.verify()


def test_canonical_bitwise_diff_partial_roi(tmp_path):
    with make_db(tmp_path) as db:
        db.add_sources([source('A',[[0,0,0],[.5,0,0],[-.5,0,0]])])
        db.correct_poses({'A':Pose.translation(.5,0,0)})
        out=db.diff((0,2,-1,1,-1,1),1,2)
        assert out['added_observations']==0
        assert out['no_longer_observed']==0
        out=db.diff((-2,3,-1,1,-1,1),1,2)
        assert out['added_observations']==1 and out['no_longer_observed']==1
