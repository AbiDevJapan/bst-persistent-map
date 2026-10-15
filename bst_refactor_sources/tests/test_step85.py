from __future__ import annotations

import inspect
import numpy as np

from synthetic.world import WorldSpec
from realdata.nclt import NCLTIngestConfig, NCLTSessionStats
from realdata.nclt_packed import PackedNCLTPrepared, linear_encode
from representations.packed_scale import PackedSnapshot, build_history_table
from representations.step83 import (
    TimeLocalAdaptiveCore, TimeLocalExplicitMap, TimeLocalTimeIndexedExplicitMap,
    SnapshotAdapter, build_session_spatial_blocks,
)
from bst_id import BSTTimeLocalMap, BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec
from benchmark.step85_benchmark import benchmark_packed_step85


def _codes(spec, coords):
    if not coords:
        return np.empty(0, dtype=np.uint64)
    a = np.asarray(sorted(coords), dtype=np.int64)
    return np.unique(linear_encode(a[:, 0], a[:, 1], a[:, 2], spec))


def _prepared5():
    spec = WorldSpec(16, 16, 2, 1)
    static = {(x, y, 0) for x in range(2, 14) for y in range(2, 14)}
    ss = []
    for t in range(5):
        cur = set(static)
        # Session-specific regions deliberately force inactive geometries into
        # the union spatial buckets, which the temporal index should skip.
        cur |= {(x, 0, 0) for x in range(2 * t, min(16, 2 * t + 4))}
        if t >= 2:
            cur -= {(6, 6, 0), (6, 7, 0), (7, 6, 0), (7, 7, 0)}
        if t == 4:
            cur |= {(15, 15, 0), (15, 14, 0)}
        ss.append(_codes(spec, cur))
    return PackedNCLTPrepared(
        spec, [f's{t}' for t in range(5)], ss, (0, 0, 0),
        [NCLTSessionStats(f's{t}') for t in range(5)],
        NCLTIngestConfig(voxel_size_m=1),
    )


def test_time_index_exact_outputs_and_same_blocks():
    p = _prepared5(); union, _ = build_history_table(p, progress=False)
    sb, _ = build_session_spatial_blocks(p, progress=False)
    core = TimeLocalAdaptiveCore(p, sb, union, progress=False)
    union_rep = TimeLocalExplicitMap(core)
    time_rep = TimeLocalTimeIndexedExplicitMap(core)
    snap = SnapshotAdapter(PackedSnapshot(p))
    regs = ((0, 8, 0, 8, 0, 2), (0, 16, 0, 16, 0, 2))
    for reg in regs:
        for t in range(5):
            ref = snap.region_query(reg, t, output='voxels')
            a = union_rep.region_query(reg, t, output='voxels')
            b = time_rep.region_query(reg, t, output='voxels')
            assert np.array_equal(a, ref)
            assert np.array_equal(b, ref)
            na = union_rep.region_query(reg, t, output='native')
            nb = time_rep.region_query(reg, t, output='native')
            assert np.array_equal(na, nb)
        for ta, tb in ((0, 1), (0, 4), (3, 1)):
            ea, er = snap.changes_query(reg, ta, tb, output='voxels')
            ga, gr = time_rep.changes_query(reg, ta, tb, output='voxels')
            assert np.array_equal(ga, ea) and np.array_equal(gr, er)


def test_time_index_reduces_or_equals_candidates_and_is_counted():
    p = _prepared5(); union, _ = build_history_table(p, progress=False)
    sb, _ = build_session_spatial_blocks(p, progress=False)
    core = TimeLocalAdaptiveCore(p, sb, union, progress=False)
    old = TimeLocalExplicitMap(core); new = TimeLocalTimeIndexedExplicitMap(core)
    reg = (0, 16, 0, 16, 0, 2)
    strict = False
    for t in range(5):
        old.region_query(reg, t, output='native'); so = dict(old.last_query_stats)
        new.region_query(reg, t, output='native'); sn = dict(new.last_query_stats)
        assert sn['blocks_scanned'] <= so['blocks_scanned']
        assert sn['bucket_refs'] <= so['bucket_refs'] or sn['blocks_scanned'] < so['blocks_scanned']
        strict |= sn['blocks_scanned'] < so['blocks_scanned']
    assert strict
    # The extra temporal selectivity is not free: its chosen index is included
    # in the method's compact-byte accounting.
    assert new.compact_bytes() == core.common_storage_bytes('time') + 16 * len(core.blocks)
    assert old.compact_bytes() == core.common_storage_bytes('union') + 16 * len(core.blocks)
    assert core.time_bucket_references >= core.session_block_instances


def test_time_index_bst_isolation_same_codec_and_same_spatial_table():
    p = _prepared5(); union, _ = build_history_table(p, progress=False)
    sb, _ = build_session_spatial_blocks(p, progress=False)
    core = TimeLocalAdaptiveCore(p, sb, union, progress=False)
    codec = PrototypeBSTSpatialCodec()
    old = BSTTimeLocalMap(core, codec)
    new = BSTTimeLocalTimeIndexedMap(core, codec)
    assert 'bst_id' not in inspect.getsource(type(core)).lower()
    # Both methods use the exact same blocks and codec; byte differences are
    # therefore solely the selected auxiliary index.
    spatial = codec.spatial_table_bytes(core.blocks, core.spec) + 4 * len(core.blocks)
    assert old.compact_bytes() - core.common_storage_bytes('union') == spatial
    assert new.compact_bytes() - core.common_storage_bytes('time') == spatial
    reg = (0, 16, 0, 16, 0, 2)
    for t in range(5):
        assert np.array_equal(old.region_query(reg, t, output='native'), new.region_query(reg, t, output='native'))


def test_step85_scaling_smoke(tmp_path):
    p = _prepared5()
    b, q, h = benchmark_packed_step85(
        p, tmp_path / 'out', scales=(2, 3, 5), query_count=1, warmups=0, repetitions=1,
        query_seed=11, targeted_fraction=1.0, include_full=False,
        output_modes=('voxels', 'native'), progress=False,
    )
    methods = {r['method'] for r in b}
    assert {'bst_time_local', 'bst_time_local_tindex', 'time_local_tindex_explicit', 'voxel_history'} <= methods
    assert {r['sessions'] for r in b} == {2, 3, 5}
    assert any(r['output_mode'] == 'native' and r['method'] == 'bst_time_local_tindex' and r['query_kind'] == 'region_at_time' for r in q)
    assert not any(r['output_mode'] == 'native' and r['method'] == 'bst_time_local_tindex' and r['query_kind'] == 'changes' for r in q)
    assert any(r['output_mode'] == 'voxels' and r['method'] == 'bst_time_local_tindex' and r['query_kind'] == 'changes' for r in q)
    assert (tmp_path / 'out' / 'scaling_build_storage.csv').exists()
    assert (tmp_path / 'out' / 'scaling_queries.csv').exists()
