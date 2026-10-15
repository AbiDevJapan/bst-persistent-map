from synthetic import SyntheticLifelongGenerator, WorldSpec
from benchmark.queries import snapshot, region_at_time, changes_between

def make_dataset(n=25, seed=42):
    return SyntheticLifelongGenerator(
        WorldSpec(100, 100, 5, 0.5), seed=seed
    ).generate(n)

def test_change_reconstruction():
    ds = make_dataset(30)
    for i in range(1, len(ds.sessions)):
        prev = set(ds.sessions[i-1].occupied)
        cur = set(ds.sessions[i].occupied)
        ch = ds.changes[i-1]
        assert (prev - set(ch.removed)) | set(ch.added) == cur

def test_determinism():
    assert make_dataset(20, 42).sessions == make_dataset(20, 42).sessions

def test_different_seed_changes_history():
    assert make_dataset(20, 1).sessions != make_dataset(20, 2).sessions

def test_snapshot():
    ds = make_dataset()
    assert snapshot(ds, 7) == ds.sessions[7].occupied

def test_region_subset():
    ds = make_dataset()
    result = region_at_time(ds, (10, 20, 10, 20, 0, 3), 10)
    assert result.issubset(ds.sessions[10].occupied)

def test_changes_between_endpoint_difference():
    ds = make_dataset()
    region = (0, 100, 0, 100, 0, 5)
    added, removed = changes_between(ds, region, 3, 17)
    a = ds.sessions[3].occupied
    b = ds.sessions[17].occupied
    assert added == b - a
    assert removed == a - b
