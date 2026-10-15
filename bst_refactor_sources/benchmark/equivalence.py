from __future__ import annotations

import random
from dataclasses import dataclass

from synthetic.generator import GeneratedDataset
from representations import SnapshotMap, OctreeTemporalMap, BSTTemporalMapPrototype


@dataclass
class EquivalenceReport:
    snapshots_checked: int = 0
    regions_checked: int = 0
    changes_checked: int = 0


def assert_equivalent(
    dataset: GeneratedDataset,
    random_queries: int = 25,
    seed: int = 123,
) -> EquivalenceReport:
    snapshot = SnapshotMap.from_dataset(dataset)
    octree = OctreeTemporalMap.from_dataset(dataset)
    bst = BSTTemporalMapPrototype.from_dataset(dataset)

    report = EquivalenceReport()

    # Deterministic snapshot checks.
    time_samples = sorted({
        0,
        min(1, len(dataset.sessions) - 1),
        len(dataset.sessions) // 2,
        len(dataset.sessions) - 1,
    })

    for t in time_samples:
        expected = snapshot.snapshot(t)
        assert octree.snapshot(t) == expected, f"Octree snapshot mismatch at t={t}"
        assert bst.snapshot(t) == expected, f"BST snapshot mismatch at t={t}"
        report.snapshots_checked += 1

    rng = random.Random(seed)
    sx = dataset.spec.size_x_m
    sy = dataset.spec.size_y_m
    sz = dataset.spec.size_z_m

    for _ in range(random_queries):
        w = rng.choice([5.0, 20.0, 40.0])
        h = rng.choice([5.0, 20.0, 40.0])
        x0 = rng.uniform(0.0, max(0.0, sx - w))
        y0 = rng.uniform(0.0, max(0.0, sy - h))
        region = (x0, x0 + w, y0, y0 + h, 0.0, sz)
        t = rng.randrange(len(dataset.sessions))

        expected = snapshot.region_at_time(region, t)
        got_oct = octree.region_at_time(region, t)
        got_bst = bst.region_at_time(region, t)

        assert got_oct == expected, f"Octree region mismatch: region={region}, t={t}"
        assert got_bst == expected, f"BST region mismatch: region={region}, t={t}"
        report.regions_checked += 1

        t_a = rng.randrange(len(dataset.sessions))
        t_b = rng.randrange(len(dataset.sessions))

        expected_change = snapshot.changes(region, t_a, t_b)
        got_oct_change = octree.changes(region, t_a, t_b)
        got_bst_change = bst.changes(region, t_a, t_b)

        assert got_oct_change == expected_change, (
            f"Octree change mismatch: region={region}, t={t_a}->{t_b}"
        )
        assert got_bst_change == expected_change, (
            f"BST change mismatch: region={region}, t={t_a}->{t_b}"
        )
        report.changes_checked += 1

    return report
