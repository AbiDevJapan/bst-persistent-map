from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import pytest
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.region_index import _author_payload

class Prepared:
    def __init__(self):
        self.spec=WorldSpec(320.,384.,48.,.5)
        self.origin_local_m=(-411.,-798.5,-47.5)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17']
        rng=np.random.default_rng(2026)
        base=np.column_stack((rng.integers(0,self.spec.nx,2800),
                              rng.integers(0,self.spec.ny,2800),
                              rng.integers(0,self.spec.nz,2800))).astype(np.uint64)
        self.session_codes=[]
        for i in range(3):
            pts=base.copy();pts[::(7+i),0]=rng.integers(0,self.spec.nx,len(pts[::(7+i)]))
            self.session_codes.append(np.unique(linear_encode(pts[:,0],pts[:,1],pts[:,2],self.spec)))

def test_actual_official_labels_and_exact_prefix_region_queries(tmp_path):
    # Import project prototype bst_id first: author's code must still resolve.
    import bst_id
    prep=Prepared();db=VersionedStore(tmp_path/'db');db.create(prep,'official',16)
    for sess in prep.session_names:
        manifest=db.all_versions()[sess]
        assert all(len(_author_payload(item['id']))==4 for item in manifest.values())
        for mode in ('bst_prefix','morton_hash'):
            result=db.build_spatial_index(sess,mode)
            assert result['tiles'] == len(manifest)
            assert result['index_disk_bytes']>0
    rng=np.random.default_rng(54)
    n=0
    for i in range(200):
        size=rng.choice([1,3,16,17,32,70,500])
        center=rng.integers([-10,-10,-5], [prep.spec.nx+10,prep.spec.ny+10,prep.spec.nz+5])
        lower=center-size//2
        upper=lower+size
        bounds=tuple(int(t) for j in range(3) for t in (lower[j],upper[j]))
        for sess in prep.session_names:
            a,base_stats=db.query(sess,bounds,stats=True)
            for mode in ('bst_prefix','morton_hash'):
                b,stats=db.query(sess,bounds,stats=True,index_mode=mode)
                np.testing.assert_array_equal(a,b)
                assert stats['tiles_read']==base_stats['tiles_read']
                assert stats['compressed_bytes_read']==base_stats['compressed_bytes_read']
            n+=1
    assert n==600


def test_stale_index_rejected_on_manifest_mutation(tmp_path):
    db=VersionedStore(tmp_path/'db');db.create(Prepared(),'official',16)
    s=Prepared().session_names[0];db.build_spatial_index(s,'bst_prefix')
    db.versions.write_text(db.versions.read_text().replace('"count":', '"old_count":',1))
    with pytest.raises(ValueError, match='stale'):
        db.query(s,(0,100,0,100,0,20),index_mode='bst_prefix')


def test_sync_differences_and_history_with_index(tmp_path):
    db=VersionedStore(tmp_path/'db');prep=Prepared();db.create(prep,'official',16)
    for s in prep.session_names:
        for mode in ('bst_prefix','morton_hash'):db.build_spatial_index(s,mode)
    roi=(31,80,50,101,0,40)
    for mode in ('bst_prefix','morton_hash'):
        expected=db.changes(prep.session_names[0],prep.session_names[2],roi)
        got=db.changes(prep.session_names[0],prep.session_names[2],roi,index_mode=mode)
        for a,b in zip(expected,got):np.testing.assert_array_equal(a,b)
        packet=db.sync_packet(prep.session_names[0],prep.session_names[1],roi,index_mode=mode)
        control=db.sync_packet(prep.session_names[0],prep.session_names[1],roi,index_mode='scan')
        assert packet==control


def test_morton_on_nonofficial_database_and_benchmark(tmp_path):
    prep=Prepared();db=VersionedStore(tmp_path/'db');db.create(prep,'morton',32)
    for name in prep.session_names:db.build_spatial_index(name,'morton_hash')
    with pytest.raises(ValueError):db.build_spatial_index(prep.session_names[0],'bst_prefix')
    from versioned_map.benchmark import run_benchmark
    rows=run_benchmark(db.root,tmp_path/'benchmark',query_count=15,repeats=1,seed=4)
    assert len(rows)==10
    assert json.loads((tmp_path/'benchmark'/'metadata.json').read_text())['query_count']==15
    assert (tmp_path/'benchmark'/'trials.csv').exists()
