from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path

from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.interval_store import IntervalMap
from versioned_map.aggregate_4d import Merged4DIndex, _merge_exact_siblings, _leaf_cells


class Prepared:
    def __init__(self):
        self.spec=WorldSpec(64,64,64,0.5)
        self.origin_local_m=(-120.,-240.,-20.)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']
        self.session_codes=[]
        # Fully populated 4 x 4 x 4 TILES with two points per tile.
        points=[]
        for tx in range(4):
            for ty in range(4):
                for tz in range(4):
                    points.append((tx*8+1,ty*8+1,tz*8+1))
                    points.append((tx*8+2,ty*8+2,tz*8+2))
        points=np.asarray(points,dtype=np.uint64)
        for s in range(5):
            p=points.copy()
            if s == 1: p[::11,0]+=2
            if s == 3: p=p[:-2]  # some partial tile observations (not tile removal)
            if s == 4: p=p[p[:,0] != 1]  # remove some full-tile presence
            self.session_codes.append(np.unique(linear_encode(*p.T,self.spec)))


@pytest.fixture
def store(tmp_path):
    prep=Prepared()
    source=VersionedStore(tmp_path/'original');source.create(prep,'official',8)
    interval=IntervalMap.build(source,tmp_path/'intervals'/'bst','bst')
    return prep,source,interval


def test_coalescence_on_each_spatial_axis_and_time():
    # Two sibling cells per axis: adding both must merge to their parent.
    for axis in range(4):
        a=[3,4,3,2,3,6,4,8]
        b=list(a);b[axis*2+1] ^= 1
        merged,metrics=_merge_exact_siblings([tuple(a),tuple(b)])
        assert len(merged)==1
        assert metrics['merged_pairs_by_axis'][('x','y','z','t')[axis]]==1
        got=next(iter(merged))
        assert got[axis*2]==a[axis*2]-1


def test_merge_preserves_all_versions_and_roi(store):
    prep,src,interval=store
    compact=Merged4DIndex(interval);report=compact.build()
    assert report['merged_four_d_cells'] < report['raw_four_d_cells'],report
    assert report['merged_pairs_by_axis']['x']>0
    assert report['merged_pairs_by_axis']['y']>0
    assert report['merged_pairs_by_axis']['z']>0
    rng=np.random.default_rng(91)
    regions=[(0,128,0,128,0,128),(-25,15,-10,10,0,32),
             (0,8,0,8,0,8),(16,32,16,32,16,32),(500,700,0,80,0,128)]
    for _ in range(40):
        center=rng.integers(-12,140,size=3);side=int(rng.choice([1,8,16,32,64,128]))
        regions.append(tuple(int(x) for d in center for x in (d,d+side)))
    for session in prep.session_names:
        for bounds in regions:
            expected=src.query(session,bounds)
            actual,info=compact.query(session,bounds,stats=True)
            np.testing.assert_array_equal(actual,expected,(session,bounds))
            assert info['replayed_transitions']==0
    assert compact.disk_stats()['cell_count']==report['merged_four_d_cells']


def test_historical_range_and_no_phantom_tiles(store):
    prep,src,interval=store
    compact=Merged4DIndex(interval);compact.build()
    bounds=(0,32,0,32,0,32)
    for a,b in ((0,0),(0,4),(1,3),(3,4)):
        expected,_=interval.history_records(prep.session_names[a],prep.session_names[b],bounds)
        got,_=compact.history_records(prep.session_names[a],prep.session_names[b],bounds)
        assert sorted((r['key'],r['start'],r['stop'],r['hash']) for r in got)==sorted(
            (r['key'],r['start'],r['stop'],r['hash']) for r in expected)


def test_author_ids_preserved_and_stale_index_rejected(store):
    prep,src,interval=store
    compact=Merged4DIndex(interval);compact.build()
    # Re-open from disk, including lossless 17-byte ID validation.
    fresh=Merged4DIndex(interval.root)
    got=fresh.query(prep.session_names[0],(0,128,0,128,0,128))
    np.testing.assert_array_equal(got,src.query(prep.session_names[0],(0,128,0,128,0,128)))
    f=interval.root/'records.json'
    f.write_text(f.read_text()+' ')
    with pytest.raises(IOError,match='stale'):
        Merged4DIndex(interval.root).load()


def test_unmerged_control_same_results(store):
    prep,src,interval=store
    obj=Merged4DIndex(interval,coalesced=False);obj.build()
    bounds=(0,128,0,128,0,128)
    for session in prep.session_names:
        np.testing.assert_array_equal(obj.query(session,bounds),src.query(session,bounds))


def test_exact_voxel_cell_aggregation_over_xyz_and_time(store,tmp_path):
    from versioned_map.voxel_4d import Voxel4D
    prep,src,_=store
    bounds=(0,32,0,32,0,32)
    comp,report=Voxel4D.build(src,tmp_path/'voxel_4d.npz',bounds,max_atoms=30000)
    assert report['final_4d_cells'] <= report['raw_4d_cells']
    opened=Voxel4D(comp.path)
    for session in prep.session_names:
        for b in (bounds,(0,8,0,8,0,8),(8,24,8,24,8,24),(-15,10,-5,40,0,16)):
            np.testing.assert_array_equal(opened.query(session,b),src.query(session,b))
    assert report['raw_4d_cells']>=report['final_4d_cells']


def test_dense_voxel_occupancy_merges_spatial_cells(tmp_path):
    from versioned_map.voxel_4d import Voxel4D
    class Dense:
        def __init__(self):
            self.spec=WorldSpec(16,16,16,.5)
            self.origin_local_m=(0.,0.,0.)
            self.session_names=['2012-01-15','2012-02-02','2012-03-17']
            p=np.array([(x,y,z) for x in range(4,12) for y in range(4,12) for z in range(4,12)],dtype=np.uint64)
            self.session_codes=[linear_encode(*p.T,self.spec) for _ in self.session_names]
    p=Dense();src=VersionedStore(tmp_path/'dense');src.create(p,'official',8)
    bounds=(0,16,0,16,0,16)
    idx,rep=Voxel4D.build(src,tmp_path/'dense4d.npz',bounds,max_atoms=100000)
    assert rep['final_4d_cells'] < rep['raw_4d_cells']/10,rep
    assert all(rep['merge_stats']['merged_pairs_by_axis'][ax]>0 for ax in ('x','y','z'))
    for session in p.session_names:
        np.testing.assert_array_equal(idx.query(session,bounds),src.query(session,bounds))


def test_aggregate_benchmark_and_cli(store,tmp_path):
    from versioned_map.aggregate_benchmark import run_aggregate_benchmark
    prep,src,interval=store
    root=interval.root.parent
    IntervalMap.build(src,root/'morton','morton')
    out=tmp_path/'report'
    report=run_aggregate_benchmark(src.root,root,out,queries=12,repeats=2,seed=987)
    assert report
    assert (out/'trials.csv').exists() and (out/'storage.json').exists()
    import subprocess,sys
    cmd=[sys.executable,'run_versioned_map.py','aggregate-query','--interval',str(interval.root),
         '--session',prep.session_names[0], '--bounds','0','128','0','128','0','128']
    parsed=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
    assert parsed['voxels']==len(src.query(prep.session_names[0],(0,128,0,128,0,128)))


def test_append_rebuilds_existing_merged_index(tmp_path):
    prep=Prepared();source=VersionedStore(tmp_path/'source');source.create(prep,'official',8)
    interval=IntervalMap.build(source,tmp_path/'interval'/'bst','bst',
                               through_session=prep.session_names[0])
    idx=Merged4DIndex(interval);idx.build()
    initial=idx.path.read_bytes()
    interval.append_from_store(source,prep.session_names[1])
    reopened=Merged4DIndex(interval.root)
    reopened.load()  # would refuse stale records.json digest
    np.testing.assert_array_equal(reopened.query(prep.session_names[1],(0,128,0,128,0,128)),
                                  source.query(prep.session_names[1],(0,128,0,128,0,128)))
    assert idx.path.read_bytes()!=initial
