from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.interval_store import IntervalMap, _dyadic_time_cover, MAX_TIME
from versioned_map.interval_benchmark import run_interval_benchmark


class Prepared:
    def __init__(self):
        self.spec=WorldSpec(128,192,32,.5)
        self.origin_local_m=(-411.,-798.5,-47.5)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']
        rng=np.random.default_rng(105)
        pts=np.column_stack((rng.integers(0,self.spec.nx,1200),
                             rng.integers(0,self.spec.ny,1200),
                             rng.integers(0,self.spec.nz,1200))).astype(np.uint64)
        self.session_codes=[]
        for i in range(len(self.session_names)):
            a=pts.copy()
            if i in (1,2,4):
                a[::(5+i),0]=rng.integers(0,self.spec.nx,len(a[::(5+i)]))
            if i==4:
                a=a[:850]
            if i==3:
                a=self._from_codes(self.session_codes[-1]).copy()
            self.session_codes.append(np.unique(linear_encode(a[:,0],a[:,1],a[:,2],self.spec)))

    def _from_codes(self,codes):
        from realdata.nclt_packed import linear_decode
        return np.stack(linear_decode(codes,self.spec),axis=1).astype(np.uint64)


@pytest.fixture
def dbs(tmp_path):
    prep=Prepared();source=VersionedStore(tmp_path/'snap');source.create(prep,'official',16)
    indexes={m:IntervalMap.build(source,tmp_path/'intervals'/m,m) for m in ('bst','morton')}
    return prep,source,indexes


def test_time_cover_exact_and_author_four_dimensional_roundtrip():
    from versioned_map.author import author_module
    Cell=author_module('region_algebra').Cell
    for start,stop in [(1,2),(42,99),(1320000000,MAX_TIME),(0,MAX_TIME),(123,123000)]:
        ranges=[]
        for z,p in _dyadic_time_cover(start,stop):
            assert 1<=z<=32
            c=Cell.from_parts(x=(28,120),y=(28,300),f=(24,70),t=(z,p))
            assert Cell.from_id(c.to_id())==c
            base=p << (32-z)
            ranges.append((base,base+(1<<(32-z))))
        assert ranges[0][0]==start and ranges[-1][1]==stop
        assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))


def test_all_versions_and_random_regions_match_exact(dbs):
    prep,source,indexes=dbs
    dims=source.meta()['dims']
    full=tuple(v for d in dims for v in (0,d))
    rng=np.random.default_rng(2026)
    regions=[full,(-20,20,-20,20,-20,20),(15,33,25,81,0,30),(1000,1200,800,900,10,20)]
    for _ in range(100):
        c=rng.integers([-20,-20,-20],np.array(dims)+20);side=int(rng.choice([1,4,32,64,128]))
        regions.append(tuple(int(v) for x in c for v in (x-side//2,x-side//2+side)))
    for session in prep.session_names:
        for bounds in regions:
            expected=source.query(session,bounds)
            for name,item in indexes.items():
                actual,stats=item.query(session,bounds,stats=True)
                np.testing.assert_array_equal(actual,expected,err_msg=f'{name} {session} {bounds}')
                assert stats['replayed_transitions']==0
    for item in indexes.values():
        assert item.validate_structure()
        assert item.disk_stats()['interval_records']>=1


def test_identical_payloads_and_version_runs(dbs):
    prep,source,indexes=dbs
    a,b=indexes['bst'],indexes['morton']
    assert a.records==b.records
    assert a.disk_stats()['blob_bytes']==b.disk_stats()['blob_bytes']
    assert a.meta['index_role']!=b.meta['index_role']
    # Sessions 2 -> 3 are deliberately identical.
    assert all(not(r['start']==3 and r['stop']==4) for r in a.records)
    # No packet/replay step is necessary for any stored snapshot query.
    assert all(r['hash'] in {v['hash'] for m in source.all_versions().values() for v in m.values()} for r in a.records)


def test_historical_4d_range_matches_reference(dbs):
    prep,source,indexes=dbs
    bounds=(0,256,0,384,0,64)
    for lo,hi in [(0,0),(0,4),(1,3),(2,4),(4,4)]:
        expected=set()
        for sess in prep.session_names[lo:hi+1]:
            expected.update(source._matching_keys(source.all_versions()[sess],bounds))
        for name,obj in indexes.items():
            rec,_=obj.history_records(prep.session_names[lo],prep.session_names[hi],bounds)
            present={r['key'] for r in rec}
            assert present==expected,(name,lo,hi)


def test_incremental_append_matches_full_build(tmp_path):
    prep=Prepared();source=VersionedStore(tmp_path/'source');source.create(prep,'official',16)
    for mode in ('bst','morton'):
        journal=IntervalMap.build(source,tmp_path/'partial'/mode,mode,through_session=prep.session_names[0])
        for sess in prep.session_names[1:]:
            report=journal.append_from_store(source,sess)
            assert report['session']==sess
            b=(0,256,0,384,0,64)
            np.testing.assert_array_equal(journal.query(sess,b),source.query(sess,b))
        complete=IntervalMap.build(source,tmp_path/'full'/mode,mode)
        assert journal.records==complete.records
        assert journal.meta['sessions']==complete.meta['sessions']
        with pytest.raises(ValueError,match='next chronological'):
            journal.append_from_store(source,prep.session_names[-1])


def test_checksum_and_index_tampering(dbs):
    prep,src,indexes=dbs
    db=indexes['bst'];item=db.records[0]
    p=db.root/'blobs'/item['hash'];p.write_bytes(b'tampered')
    with pytest.raises(IOError,match='checksum'):
        db.query(prep.session_names[item['start']],(0,256,0,384,0,64))
    db=indexes['morton'];p=db.root/'records.json';p.write_text(p.read_text()+' ')
    with pytest.raises(IOError,match='stale'):
        IntervalMap(db.root).query(prep.session_names[0],(0,256,0,384,0,64))


def test_benchmark_and_cli(dbs,tmp_path):
    prep,source,idx=dbs
    out=tmp_path/'benchmark'
    stats=run_interval_benchmark(source.root,tmp_path/'intervals',out,queries=15,repeats=2,seed=77,verify_full=True)
    assert (out/'trials.csv').exists() and (out/'storage.json').exists()
    assert len(stats)>10
    r=subprocess.run([sys.executable,'run_versioned_map.py','interval-query',
                      '--interval',str(idx['bst'].root),'--session',prep.session_names[0],
                      '--bounds','0','256','0','384','0','64'],capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['replayed_transitions']==0
