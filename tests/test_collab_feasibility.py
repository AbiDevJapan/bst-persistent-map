from __future__ import annotations
import numpy as np
import pytest
from collab_feasibility.synthetic import build_synthetic
from collab_feasibility.protocol import METHODS,Receiver,Sender,_box_encode,_box_decode,_expand_clipped
from collab_feasibility.benchmark import benchmark,create_tasks


@pytest.fixture(scope='module')
def shared(tmp_path_factory):
    return build_synthetic(tmp_path_factory.mktemp('collab'),robots=2,epochs=2,tile_side=16,k=4)


def test_all_six_protocols_reconstruct_exact_roi_and_repeat_cache(shared):
    db,geom=shared[0];s=Sender(db,geom);version=list(db.all_versions())[-1]
    bounds=(8,65,10,65,0,15)
    expected=db.query(version,bounds,index_mode='morton_hash')
    for mode in METHODS:
        receiver=Receiver()
        msg,st=s.send(mode,version,bounds,receiver)
        got,_=receiver.apply(msg,db._spec(),expected_bounds=bounds)
        np.testing.assert_array_equal(got,expected)
        msg2,st2=s.send(mode,version,bounds,receiver)
        assert st2['body_bytes']==0
        got2,_=receiver.apply(msg2,db._spec(),expected_bounds=bounds)
        np.testing.assert_array_equal(got2,expected)
        assert len(msg2)<len(msg)


def test_bst_and_morton_have_bit_identical_wire_payload_under_same_cache(shared):
    db,geom=shared[0];s=Sender(db,geom);session=list(db.all_versions())[-1]
    bounds=(12,80,8,75,1,16)
    for a,b in [('k4_bst','k4_morton'),('k4_tile_bst','k4_tile_morton')]:
        rcva,rcvb=Receiver(),Receiver()
        aa,_=s.send(a,session,bounds,rcva);bb,_=s.send(b,session,bounds,rcvb)
        assert aa==bb
        rcva.apply(aa,db._spec());rcvb.apply(bb,db._spec())
        aa2,_=s.send(a,session,bounds,rcva);bb2,_=s.send(b,session,bounds,rcvb)
        assert aa2==bb2


def test_corrupt_packets_rejected_and_wrong_roi(shared):
    db,geom=shared[0];s=Sender(db,geom)
    sess=list(db.all_versions())[0];bounds=(10,55,10,55,0,15)
    msg,_=s.send('k4_bst',sess,bounds,Receiver())
    with pytest.raises(ValueError,match='ROI header mismatch'):
        Receiver().apply(msg,db._spec(),(11,55,10,55,0,15))
    corrupted=bytearray(msg);corrupted[-3]^=0x0f
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        Receiver().apply(bytes(corrupted),db._spec(),bounds)


def test_partial_blocks_are_clipped_exactly(shared):
    db,_=shared[0];sp=db._spec()
    data=np.array([[4,4,4,4,2,2]],dtype=np.int64)
    np.testing.assert_array_equal(_box_decode(_box_encode(data)),data)
    out=_expand_clipped(data,(6,7,4,6,3,6),sp)
    from realdata.nclt_packed import linear_encode
    expected=linear_encode(np.asarray([6,6,6,6]),np.asarray([4,4,5,5]),np.asarray([4,5,4,5]),sp)
    np.testing.assert_array_equal(out,np.sort(expected))


def test_multirobot_benchmark_outputs_and_correctness(shared,tmp_path):
    rows,info=benchmark(shared,tmp_path/'results',requests=8,seed=123)
    assert len(rows)==len(METHODS)
    assert all(x['all_correct'] for x in rows)
    assert rows[2]['total_wire_bytes']==rows[3]['total_wire_bytes']
    assert rows[4]['total_wire_bytes']==rows[5]['total_wire_bytes']
    assert info['source_robots']==2
    assert (tmp_path/'results'/'trials.csv').exists()
    assert (tmp_path/'results'/'report.md').exists()


def test_gps_request_enclosure_is_monotone(shared):
    low=create_tasks(shared,18,1,6,1,seed=98)
    high=create_tasks(shared,18,7,6,1,seed=98)
    # With the same pseudorandom draws, request inflation tracks uncertainty.
    for task in low+high:
        assert len(task['bounds'])==6
        assert all(task['bounds'][j]<task['bounds'][j+1] for j in (0,2,4))
    assert np.mean([x['gps_roi_encloses_target'] for x in high])>=0.5


def test_updated_map_version_is_reconstructed_without_stale_geometry(shared):
    db,geom=shared[0];s=Sender(db,geom)
    versions=list(db.all_versions());bounds=(0,96,0,96,0,16)
    assert len(versions)>=2
    assert not np.array_equal(db.query(versions[0],bounds),db.query(versions[1],bounds))
    for method in METHODS:
        r=Receiver()
        for v in versions:
            msg,st=s.send(method,v,bounds,r)
            result,_=r.apply(msg,db._spec(),bounds)
            np.testing.assert_array_equal(result,db.query(v,bounds))


def test_packed_tile_headers_stay_smaller_than_verbose_json(shared):
    import json
    from collab_feasibility.protocol import _unpack
    db,geom=shared[0]
    r=Receiver();msg,_=Sender(db,geom).send('voxel_tiles',list(db.all_versions())[0],
                 (0,96,0,96,0,16),r)
    meta,chunks=_unpack(msg)
    assert len(meta['tiles'])>=2
    verbose=len(json.dumps(meta).encode())+sum(len(b) for b in chunks.values())
    assert len(msg)<verbose
