from pathlib import Path
import argparse
import time

from synthetic import SyntheticLifelongGenerator, WorldSpec
from benchmark.queries import snapshot, region_at_time, changes_between, change_frequency_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    spec = WorldSpec(100.0, 100.0, 5.0, 0.5)

    t0 = time.perf_counter()
    dataset = SyntheticLifelongGenerator(spec, args.seed).generate(args.sessions)
    elapsed = time.perf_counter() - t0

    print(f"Generated {len(dataset.sessions)} sessions in {elapsed:.3f} s")
    print(f"Voxel grid: {spec.nx} x {spec.ny} x {spec.nz}")
    print(f"Session 0 occupied voxels: {len(dataset.sessions[0].occupied):,}")
    print(f"Final occupied voxels:     {len(dataset.sessions[-1].occupied):,}")

    if dataset.changes:
        mean_change = sum(c.changed_count for c in dataset.changes) / len(dataset.changes)
        print(f"Mean changed voxels/session: {mean_change:,.1f}")

    t = min(37, len(dataset.sessions)-1)
    small = (30, 35, 60, 65, 0, 3)
    medium = (20, 40, 50, 70, 0, 5)

    print(f"\nSnapshot({t}): {len(snapshot(dataset, t)):,} occupied voxels")
    print(f"Small region @ {t}: {len(region_at_time(dataset, small, t)):,} occupied voxels")

    t_a = min(20, len(dataset.sessions)-1)
    t_b = min(40, len(dataset.sessions)-1)
    added, removed = changes_between(dataset, medium, t_a, t_b)
    print(f"Medium-region changes {t_a}->{t_b}: +{len(added):,}, -{len(removed):,}")

    print("\nMost frequently changed voxels:")
    for voxel, count in change_frequency_map(dataset).most_common(5):
        print(f"  {voxel}: {count} transitions")

    if args.save:
        dataset.save(Path(args.save))
        print(f"\nSaved dataset to {Path(args.save).resolve()}")

if __name__ == "__main__":
    main()
