from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple
import random

from .world import Voxel, WorldSpec

Bounds2D = Tuple[float, float, float, float]

@dataclass(frozen=True)
class CuboidObject:
    object_id: str
    category: str
    zone: str
    center_m: Tuple[float, float, float]
    size_m: Tuple[float, float, float]
    allowed_xy_m: Bounds2D
    active: bool = True

    def voxels(self, spec: WorldSpec) -> set[Voxel]:
        if not self.active:
            return set()

        cx, cy, cz = self.center_m
        sx, sy, sz = self.size_m
        r = spec.voxel_size_m

        x0 = max(0, int((cx - sx / 2) // r))
        x1 = min(spec.nx - 1, int((cx + sx / 2) // r))
        y0 = max(0, int((cy - sy / 2) // r))
        y1 = min(spec.ny - 1, int((cy + sy / 2) // r))
        z0 = max(0, int((cz - sz / 2) // r))
        z1 = min(spec.nz - 1, int((cz + sz / 2) // r))

        return {
            (x, y, z)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
            for z in range(z0, z1 + 1)
            if spec.contains_voxel((x, y, z))
        }

    def moved(self, rng: random.Random) -> "CuboidObject":
        xmin, xmax, ymin, ymax = self.allowed_xy_m
        sx, sy, _ = self.size_m
        margin_x = sx / 2 + 0.5
        margin_y = sy / 2 + 0.5

        lo_x, hi_x = xmin + margin_x, xmax - margin_x
        lo_y, hi_y = ymin + margin_y, ymax - margin_y
        if lo_x >= hi_x or lo_y >= hi_y:
            return self

        _, _, cz = self.center_m
        return replace(
            self,
            center_m=(rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y), cz),
        )

    def toggled(self) -> "CuboidObject":
        return replace(self, active=not self.active)
