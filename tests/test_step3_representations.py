import random

from synthetic import SyntheticLifelongGenerator, WorldSpec
from representations import (
    SnapshotMap,
    OctreeTemporalMap,
    BSTTemporalMapPrototype,
    LocalMetricBSTEncoder,
)
from benchmark.equivalence import assert_equivalent


def make_dataset(n=20, seed=42):
    return SyntheticLifelongGenerator(
        WorldSpec(100, 100, 5, 0.5),
        seed=seed,
    ).generate(n)


def test_bst_encoder_round_trip():
    enc = LocalMetricBSTEncoder(nx=200, ny=200, nz=10, num_sessions=100)
    samples = [
        (0, 0, 0, 0),
        (199, 199, 9, 99),
        (123, 17, 4, 51),
        (42, 88, 1, 7),
    ]
    for item in samples:
        key = enc.encode(*item)
        assert enc.decode(key) == item


def test_bst_prefix_contains_original_point():
    enc = LocalMetricBSTEncoder(nx=200, ny=200, nz=10, num_sessions=100)
    point = (123, 17, 4, 51)
    key = enc.encode(*point)

    for prefix_len in [0, 1, 4, 8, 12, enc.total_bits]:
        prefix = enc.prefix(key, prefix_len)
        bounds = enc.prefix_bounds(prefix, prefix_len)
        x, y, z, t = point
        assert bounds.x[0] <= x <= bounds.x[1]
        assert bounds.y[0] <= y <= bounds.y[1]
        assert bounds.z[0] <= z <= bounds.z[1]
        assert bounds.t[0] <= t <= bounds.t[1]


def test_all_three_snapshot_equivalence():
    ds = make_dataset(20)
    a = SnapshotMap.from_dataset(ds)
    b = OctreeTemporalMap.from_dataset(ds)
    c = BSTTemporalMapPrototype.from_dataset(ds)

    for t in [0, 1, 5, 10, 19]:
        assert a.snapshot(t) == b.snapshot(t) == c.snapshot(t)


def test_all_three_region_equivalence():
    ds = make_dataset(20)
    a = SnapshotMap.from_dataset(ds)
    b = OctreeTemporalMap.from_dataset(ds)
    c = BSTTemporalMapPrototype.from_dataset(ds)

    regions = [
        (0, 5, 0, 5, 0, 3),
        (10, 30, 10, 30, 0, 5),
        (60, 95, 5, 60, 0, 5),
        (45, 95, 58, 94, 0, 5),
    ]
    for t in [0, 7, 19]:
        for region in regions:
            expected = a.region_at_time(region, t)
            assert b.region_at_time(region, t) == expected
            assert c.region_at_time(region, t) == expected


def test_all_three_change_equivalence():
    ds = make_dataset(20)
    a = SnapshotMap.from_dataset(ds)
    b = OctreeTemporalMap.from_dataset(ds)
    c = BSTTemporalMapPrototype.from_dataset(ds)

    region = (45, 95, 5, 95, 0, 5)
    for t_a, t_b in [(0, 1), (1, 10), (3, 19), (19, 5)]:
        expected = a.changes(region, t_a, t_b)
        assert b.changes(region, t_a, t_b) == expected
        assert c.changes(region, t_a, t_b) == expected


def test_random_equivalence_harness():
    ds = make_dataset(18)
    report = assert_equivalent(ds, random_queries=10, seed=999)
    assert report.snapshots_checked >= 3
    assert report.regions_checked == 10
    assert report.changes_checked == 10


def test_octree_and_bst_store_transitions_not_full_snapshots():
    ds = make_dataset(20)
    snapshots = SnapshotMap.from_dataset(ds)
    octree = OctreeTemporalMap.from_dataset(ds)
    bst = BSTTemporalMapPrototype.from_dataset(ds)

    full_snapshot_records = snapshots.stats()["stored_voxel_records"]
    assert octree.stats()["history_entries"] < full_snapshot_records
    assert bst.stats()["events"] < full_snapshot_records
