from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.delta_journal import DeltaJournal, source_storage_controls
from versioned_map.delta_benchmark import verify_against_store, run_delta_benchmark


class Prepared:
    def __init__(self):
        self.spec=WorldSpec(64.,64.,16.,1.)
        self.origin_local_m=(-8.,-16.,-4.)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17','2012-05-11']
        samples=[
            [(1,1,1),(2,1,1),(3,3,1),(34,35,2),(50,50,5)],
            [(1,1,1),(2,1,1),(4,4,1),(34,35,2),(50,50,5)],
            [(1,1,1),(2,1,1),(4,4,1),(31,31,0),(48,50,5)],
            [(1,1,1),(2,1,1),(4,4,1),(31,31,0),(48,50,5)],
        ]
        self.session_codes=[]
        for points in samples:
            a=np.asarray(points, dtype=np.uint64)
            self.session_codes.append(np.unique(linear_encode(a[:,0],a[:,1],a[:,2],self.spec)))


@pytest.fixture
def built(tmp_path):
    prep=Prepared();store=VersionedStore(tmp_path/'snapshots');store.create(prep,'official',16)
    journals={mode:DeltaJournal.build(store,tmp_path/'journals'/mode,mode) for mode in ('morton','bst')}
    return prep,store,journals


def test_all_full_versions_exact_and_storage(built):
    prep,store,journals=built
    for mode,j in journals.items():
        assert j.validate_structure()
        for sess,canonical in zip(prep.session_names,prep.session_codes):
            np.testing.assert_array_equal(j.reconstruct(sess),canonical)
        s=j.disk_stats()
        assert s['unique_blob_count']>0 and s['total_bytes']>s['unique_blob_bytes']
        assert len(j.meta['transitions'])==3
        assert j.meta['transitions'][-1]['changed_tiles']==0
        assert j.meta['transitions'][-1]['compressed_transition_payload_bytes_repeated']==0
        assert all('address' in r for r in j.base.values())
    assert source_storage_controls(store)['independent_total_bytes_estimate']>0


def test_repeated_random_roi_exact_all_historical_sessions(built):
    prep,store,journals=built
    bounds=[(-100,5,-10,4,0,3),(0,64,0,64,0,16),(32,48,32,48,0,16),
            (16,32,16,32,0,4),(100,120,100,120,50,60)]
    rng=np.random.default_rng(555)
    for _ in range(50):
        start=rng.integers([-20,-20,-5],[70,70,25]);width=rng.integers(1,40,size=3)
        bounds.append(tuple(int(x) for j in range(3) for x in (start[j],start[j]+width[j])))
    for session in prep.session_names:
        for b in bounds:
            control=store.query(session,b)
            for mode in ('morton','bst'):
                actual,stat=journals[mode].query(session,b,stats=True)
                np.testing.assert_array_equal(actual,control)
                assert stat['candidate_tiles']>=0
    verify_against_store(store,journals['bst'],random_queries=30,seed=37,full=True)


def test_payloads_match_but_address_labels_differ(built):
    _,_,journals=built
    a,b=journals['morton'],journals['bst']
    assert set(a.base)==set(b.base)
    for key in a.base:
        assert a.base[key]['hash']==b.base[key]['hash']
        assert a.base[key]['address']!=b.base[key]['address']
    for left,right in zip(a.transitions,b.transitions):
        assert set(left)==set(right)
        for key in left:
            for field in ('hash','add','remove'):
                assert left[key][field]==right[key][field]
            assert left[key]['address']!=right[key]['address']
    assert all(j.meta['address_role']=='journal labels only; spatial lookup uses canonical tile coordinates'
               for j in journals.values())


def test_corruption_detection_and_mutated_snapshots(tmp_path,built):
    prep,store,journals=built
    from versioned_map.region_index import manifest_hash
    journal=journals['morton']
    store.versions.write_text(store.versions.read_text().replace('"count":','"bad_count":',1))
    with pytest.raises(ValueError,match='source manifest changed'):
        verify_against_store(store,journal,random_queries=1)
    item=next(iter(journal.transitions[0].values()))
    (journal.root/'blobs'/item['hash']).write_bytes(b'tampered')
    with pytest.raises(IOError,match='hash'):
        journal.reconstruct(prep.session_names[1])


def test_benchmark_files_and_cli(tmp_path,built):
    prep,store,journals=built
    out=tmp_path/'performance'
    summary=run_delta_benchmark(store.root,tmp_path/'journals',out,queries=12,repeats=2,seed=61)
    assert len(summary)==12
    assert (out/'trials.csv').exists() and (out/'summary.csv').exists()
    assert (out/'storage.json').exists() and (out/'metadata.json').exists()
    assert json.loads((out/'metadata.json').read_text())['queries']==12
    subprocess.run([sys.executable,'run_versioned_map.py','delta-query','--journal',str(journals['bst'].root),
                    '--session',prep.session_names[0], '--bounds','0','10','0','10','0','10'],
                   check=True,capture_output=True,text=True)


def test_refuses_rebuilding_and_incompatible_codec(tmp_path,built):
    prep,store,journals=built
    with pytest.raises(FileExistsError):
        DeltaJournal.build(store,journals['bst'].root,'bst')
    with pytest.raises(ValueError):
        DeltaJournal.build(store,tmp_path/'wrong','unknown')
    old=VersionedStore(tmp_path/'prototype');old.create(prep,'prototype',16)
    with pytest.raises(ValueError,match='official'):
        DeltaJournal.build(old,tmp_path/'bad','bst')


def test_incremental_append_one_session_at_a_time(tmp_path):
    prep=Prepared();store=VersionedStore(tmp_path/'db');store.create(prep,'official',16)
    for mode in ('morton','bst'):
        j=DeltaJournal.build(store,tmp_path/'journal'/mode,mode,through_session=prep.session_names[0])
        assert j.meta['sessions']==prep.session_names[:1]
        for session in prep.session_names[1:]:
            update=j.append_from_store(store,session)
            assert update['to']==session
            np.testing.assert_array_equal(store.query(session,(0,64,0,64,0,16)),j.reconstruct(session))
        assert j.meta['sessions']==prep.session_names
        assert len(j.transitions)==3
        with pytest.raises(ValueError,match='next chronological'):
            j.append_from_store(store,prep.session_names[-1])
