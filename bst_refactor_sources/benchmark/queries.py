from __future__ import annotations
from collections import Counter
from typing import Tuple

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel

MetricBox = Tuple[float, float, float, float, float, float]

def snapshot(dataset: GeneratedDataset, t: int) -> frozenset[Voxel]:
    if t < 0 or t >= len(dataset.sessions):
        raise IndexError(f"session {t} outside valid range")
    return dataset.sessions[t].occupied

def _bounds(dataset: GeneratedDataset, region: MetricBox):
    xmin, xmax, ymin, ymax, zmin, zmax = region
    r = dataset.spec.voxel_size_m
    return (
        max(0, int(xmin // r)),
        min(dataset.spec.nx - 1, int(xmax // r)),
        max(0, int(ymin // r)),
        min(dataset.spec.ny - 1, int(ymax // r)),
        max(0, int(zmin // r)),
        min(dataset.spec.nz - 1, int(zmax // r)),
    )

def region_at_time(dataset: GeneratedDataset, region: MetricBox, t: int):
    ix0, ix1, iy0, iy1, iz0, iz1 = _bounds(dataset, region)
    return frozenset(
        (x, y, z)
        for x, y, z in snapshot(dataset, t)
        if ix0 <= x <= ix1
        and iy0 <= y <= iy1
        and iz0 <= z <= iz1
    )

def changes_between(dataset: GeneratedDataset, region: MetricBox, t_a: int, t_b: int):
    if t_b < t_a:
        t_a, t_b = t_b, t_a

    a = region_at_time(dataset, region, t_a)
    b = region_at_time(dataset, region, t_b)
    return frozenset(b - a), frozenset(a - b)

def change_frequency_map(dataset: GeneratedDataset) -> Counter:
    counts = Counter()
    for change in dataset.changes:
        counts.update(change.added)
        counts.update(change.removed)
    return counts
