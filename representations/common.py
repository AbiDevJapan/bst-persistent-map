from __future__ import annotations

from typing import Tuple

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel

MetricBox = Tuple[float, float, float, float, float, float]


def metric_box_to_voxel_bounds(
    dataset: GeneratedDataset,
    region: MetricBox,
) -> tuple[int, int, int, int, int, int]:
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


def voxel_in_bounds(
    voxel: Voxel,
    bounds: tuple[int, int, int, int, int, int],
) -> bool:
    x, y, z = voxel
    x0, x1, y0, y1, z0, z1 = bounds
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1
