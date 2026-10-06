import argparse
import time

from synthetic import SyntheticLifelongGenerator, WorldSpec
from representations import SnapshotMap, OctreeTemporalMap, BSTTemporalMapPrototype
from benchmark.equivalence import assert_equivalent


def timed_build(cls, dataset):
    t0 = time.perf_counter()
    obj = cls.from_dataset(dataset)
    return obj, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-queries", type=int, default=25)
    args = parser.parse_args()

    dataset = SyntheticLifelongGenerator(
        WorldSpec(100.0, 100.0, 5.0, 0.5),
        seed=args.seed,
    ).generate(args.sessions)

    print(f"Dataset: {len(dataset.sessions)} sessions")
    print(f"Initial occupied voxels: {len(dataset.sessions[0].occupied):,}")
    print()

    snapshot, ts = timed_build(SnapshotMap, dataset)
    octree, to = timed_build(OctreeTemporalMap, dataset)
    bst, tb = timed_build(BSTTemporalMapPrototype, dataset)

    print("Build diagnostics (NOT publication benchmarks yet)")
    print(f"  Snapshot: {ts:.4f}s  {snapshot.stats()}")
    print(f"  Octree:   {to:.4f}s  {octree.stats()}")
    print(f"  BST proto:{tb:.4f}s  {bst.stats()}")
    print()

    report = assert_equivalent(
        dataset,
        random_queries=args.random_queries,
        seed=123,
    )
    print("EQUIVALENCE PASSED")
    print(f"  snapshots: {report.snapshots_checked}")
    print(f"  regions:   {report.regions_checked}")
    print(f"  changes:   {report.changes_checked}")

    # Show one diagnostic query.
    region = (62.0, 82.0, 8.0, 28.0, 0.0, 5.0)
    t = min(20, args.sessions - 1)
    reference = snapshot.region_at_time(region, t)
    got_octree = octree.region_at_time(region, t)
    got_bst = bst.region_at_time(region, t)

    assert reference == got_octree == got_bst
    print()
    print(f"Example region @ t={t}: {len(reference):,} occupied voxels")
    print(f"  Octree nodes visited: {octree.last_query_nodes_visited:,}")
    print(f"  BST query diagnostics: {bst.last_query_stats}")

    print()
    print("Important: BSTTemporalMapPrototype uses a local-metric hierarchical")
    print("x/y/z/t encoder for Step-3 correctness. It is NOT yet a claim that")
    print("the encoding is byte-for-byte identical to the upstream BST-ID spec.")


if __name__ == "__main__":
    main()
