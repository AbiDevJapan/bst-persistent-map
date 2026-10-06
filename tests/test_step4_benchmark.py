from benchmark.step4_benchmark import run_benchmark
from benchmark.storage import compact_storage_bytes
from synthetic import SyntheticLifelongGenerator, WorldSpec
from representations import SnapshotMap, OctreeTemporalMap, BSTTemporalMapPrototype

def test_storage_metrics():
    ds=SyntheticLifelongGenerator(WorldSpec(100,100,5,0.5),42).generate(10)
    reps=[SnapshotMap.from_dataset(ds),OctreeTemporalMap.from_dataset(ds),BSTTemporalMapPrototype.from_dataset(ds)]
    sizes=[compact_storage_bytes(x) for x in reps]
    assert all(x>0 for x in sizes); assert sizes[0]>sizes[2]

def test_quick_benchmark(tmp_path):
    build,query=run_benchmark(tmp_path/'out',[6,8],query_count=1,warmups=0,repetitions=1)
    assert len(build)==6 and query
    assert (tmp_path/'out'/'build_storage.csv').exists(); assert (tmp_path/'out'/'queries.csv').exists(); assert (tmp_path/'out'/'metadata.json').exists()
