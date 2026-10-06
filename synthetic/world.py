from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, Tuple

Voxel = Tuple[int, int, int]

@dataclass(frozen=True)
class WorldSpec:
    size_x_m: float = 100.0
    size_y_m: float = 100.0
    size_z_m: float = 5.0
    voxel_size_m: float = 0.5

    @property
    def nx(self) -> int:
        return round(self.size_x_m / self.voxel_size_m)

    @property
    def ny(self) -> int:
        return round(self.size_y_m / self.voxel_size_m)

    @property
    def nz(self) -> int:
        return round(self.size_z_m / self.voxel_size_m)

    def contains_voxel(self, v: Voxel) -> bool:
        x, y, z = v
        return 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz

@dataclass(frozen=True)
class Session:
    session_id: int
    occupied: FrozenSet[Voxel]

@dataclass(frozen=True)
class Change:
    session_id: int
    added: FrozenSet[Voxel]
    removed: FrozenSet[Voxel]

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.removed)
