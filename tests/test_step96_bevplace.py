import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization.step96_bevplace import BEVPlaceDatabase, BEVPlaceDescriptorCache, unit_descriptors


def _write(path, files, indices, descriptors, sha='123', preprocess='bev_v1', **extras):
    np.savez(path, descriptors=np.asarray(descriptors,dtype=np.float32),scan_files=np.array(files),
        original_scan_indices=np.asarray(indices,dtype=np.int64),
        checkpoint_sha256=np.array(sha),preprocess_id=np.array(preprocess),model_id=np.array('REIN'),**extras)


def test_nearest_neighbour_and_tie_breaking():
    db=BEVPlaceDatabase([[1,0],[0,1],[-1,0],[1,0]])
    ranked=db.query_descriptor([2,0],top_k=4)
    assert [m.reference_index for m in ranked]==[0,3,1,2]
    assert ranked[0].distance == 0.0


def test_unit_rejects_poisoned_or_degenerate():
    with pytest.raises(ValueError): unit_descriptors([[np.nan,1]])
    with pytest.raises(ValueError): unit_descriptors([[0,0]])


def test_exact_reference_order_and_query_mapping(tmp_path):
    _write(tmp_path/'reference_jan.npz',['a.bin','b.bin'],[10,20],[[1,0],[0,1]],pose_rows=np.zeros((2,7)))
    _write(tmp_path/'query_feb.npz',['q.bin'],[33],[[0.9,.1]])
    cache=BEVPlaceDescriptorCache(tmp_path)
    db,q=cache.load_pair('jan','feb',np.array([10,20]),['a.bin','b.bin'])
    desc=cache.query_descriptor(q['query_by_name'],'q.bin',33)
    assert db.query_descriptor(desc)[0].reference_index==0
    with pytest.raises(ValueError,match='index mismatch'):
        cache.query_descriptor(q['query_by_name'],'q.bin',32)
    with pytest.raises(ValueError,match='EXACT'):
        cache.load_pair('jan','feb',np.array([20,10]),['b.bin','a.bin'])


def test_mismatched_model_checkpoint_rejected(tmp_path):
    _write(tmp_path/'reference_jan.npz',['a.bin'],[10],[[1,0]],sha='abc')
    _write(tmp_path/'query_feb.npz',['q.bin'],[20],[[1,0]],sha='different')
    with pytest.raises(ValueError,match='metadata mismatch'):
        BEVPlaceDescriptorCache(tmp_path).load_pair('jan','feb',[10],['a.bin'])
