from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import random

import numpy as np

from .objects import CuboidObject
from .world import Change, Session, Voxel, WorldSpec

ZONE_CHANGE_PROB = {
    "static": 0.0,
    "slow": 0.03,
    "medium": 0.15,
    "high": 0.75,
}

@dataclass
class GeneratedDataset:
    spec: WorldSpec
    sessions: List[Session]
    changes: List[Change]
    objects_by_session: List[Dict[str, CuboidObject]]

    def validate(self) -> None:
        assert self.sessions
        assert self.sessions[0].session_id == 0

        for i in range(1, len(self.sessions)):
            prev = set(self.sessions[i - 1].occupied)
            cur = set(self.sessions[i].occupied)
            ch = self.changes[i - 1]
            reconstructed = (prev - set(ch.removed)) | set(ch.added)

            assert reconstructed == cur, f"reconstruction failed at session {i}"
            assert set(ch.added) == cur - prev
            assert set(ch.removed) == prev - cur

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        (output_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (output_dir / "changes").mkdir(parents=True, exist_ok=True)

        metadata = {
            "world": {
                "size_x_m": self.spec.size_x_m,
                "size_y_m": self.spec.size_y_m,
                "size_z_m": self.spec.size_z_m,
                "voxel_size_m": self.spec.voxel_size_m,
                "nx": self.spec.nx,
                "ny": self.spec.ny,
                "nz": self.spec.nz,
            },
            "num_sessions": len(self.sessions),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        for session in self.sessions:
            arr = np.asarray(sorted(session.occupied), dtype=np.uint16)
            np.savez_compressed(
                output_dir / "sessions" / f"session_{session.session_id:04d}.npz",
                occupied=arr,
            )

        for change in self.changes:
            added = np.asarray(sorted(change.added), dtype=np.uint16)
            removed = np.asarray(sorted(change.removed), dtype=np.uint16)
            np.savez_compressed(
                output_dir / "changes" / f"change_{change.session_id:04d}.npz",
                added=added,
                removed=removed,
            )

class SyntheticLifelongGenerator:
    def __init__(
        self,
        spec: WorldSpec | None = None,
        seed: int = 42,
        change_prob: dict[str, float] | None = None,
    ):
        self.spec = spec or WorldSpec()
        self.seed = seed
        self.change_prob = dict(
            ZONE_CHANGE_PROB if change_prob is None else change_prob
        )
        self.rng = random.Random(seed)

    def _wall_box(self, object_id, center, size, area, category="wall"):
        return CuboidObject(
            object_id=object_id,
            category=category,
            zone="static",
            center_m=center,
            size_m=size,
            allowed_xy_m=area,
        )

    def _make_static_objects(self) -> list[CuboidObject]:
        objs: list[CuboidObject] = []
        wall_h = 3.0
        thickness = 0.5

        # Building A.
        x0, x1, y0, y1 = 5.0, 40.0, 8.0, 38.0
        area = (x0, x1, y0, y1)
        objs += [
            self._wall_box("A_north", ((x0+x1)/2, y1, wall_h/2), (x1-x0, thickness, wall_h), area),
            self._wall_box("A_south", ((x0+x1)/2, y0, wall_h/2), (x1-x0, thickness, wall_h), area),
            self._wall_box("A_west", (x0, (y0+y1)/2, wall_h/2), (thickness, y1-y0, wall_h), area),
            self._wall_box("A_east", (x1, (y0+y1)/2, wall_h/2), (thickness, y1-y0, wall_h), area),
            self._wall_box("A_inner_1", (22.5, 18.0, wall_h/2), (25.0, thickness, wall_h), area),
            self._wall_box("A_inner_2", (22.5, 28.0, wall_h/2), (25.0, thickness, wall_h), area),
            self._wall_box("A_corridor", (18.0, 23.0, wall_h/2), (thickness, 10.0, wall_h), area),
        ]

        # Warehouse.
        x0, x1, y0, y1 = 8.0, 38.0, 60.0, 88.0
        area = (x0, x1, y0, y1)
        objs += [
            self._wall_box("W_north", ((x0+x1)/2, y1, wall_h/2), (x1-x0, thickness, wall_h), area),
            self._wall_box("W_south", ((x0+x1)/2, y0, wall_h/2), (x1-x0, thickness, wall_h), area),
            self._wall_box("W_west", (x0, (y0+y1)/2, wall_h/2), (thickness, y1-y0, wall_h), area),
            self._wall_box("W_east", (x1, (y0+y1)/2, wall_h/2), (thickness, y1-y0, wall_h), area),
        ]

        # Road-side poles.
        for i, (x, y) in enumerate(
            [(55, 10), (55, 25), (55, 40), (55, 55), (55, 70), (55, 85)]
        ):
            objs.append(
                CuboidObject(
                    object_id=f"pole_{i}",
                    category="pole",
                    zone="static",
                    center_m=(x, y, 1.5),
                    size_m=(0.5, 0.5, 3.0),
                    allowed_xy_m=(x-1, x+1, y-1, y+1),
                )
            )
        return objs

    def _random_object(self, object_id, category, zone, size_m, bounds):
        xmin, xmax, ymin, ymax = bounds
        sx, sy, sz = size_m
        cx = self.rng.uniform(xmin + sx, xmax - sx)
        cy = self.rng.uniform(ymin + sy, ymax - sy)
        cz = max(sz / 2.0, 0.25)

        return CuboidObject(
            object_id=object_id,
            category=category,
            zone=zone,
            center_m=(cx, cy, cz),
            size_m=size_m,
            allowed_xy_m=bounds,
        )

    def _make_dynamic_objects(self) -> list[CuboidObject]:
        objs: list[CuboidObject] = []

        # Slow-changing furniture / equipment.
        slow_bounds = [(10, 35, 10, 36), (12, 35, 63, 85)]
        for i in range(18):
            objs.append(
                self._random_object(
                    f"slow_{i}",
                    "equipment",
                    "slow",
                    (1.5, 1.0, 1.0),
                    slow_bounds[i % 2],
                )
            )

        # Medium-change parked cars.
        for i in range(20):
            objs.append(
                self._random_object(
                    f"medium_{i}",
                    "parked_car",
                    "medium",
                    (4.5, 2.0, 1.5),
                    (62, 95, 8, 55),
                )
            )

        # Highly dynamic temporary clutter.
        for i in range(30):
            objs.append(
                self._random_object(
                    f"high_{i}",
                    "temporary_box",
                    "high",
                    (1.0, 1.0, 1.0),
                    (45, 95, 58, 94),
                )
            )
        return objs

    def _all_voxels(self, objects: Dict[str, CuboidObject]) -> frozenset[Voxel]:
        occupied: set[Voxel] = set()
        for obj in objects.values():
            occupied |= obj.voxels(self.spec)
        return frozenset(occupied)

    def _advance_object(self, obj: CuboidObject) -> CuboidObject:
        p = self.change_prob[obj.zone]
        if p <= 0.0 or self.rng.random() >= p:
            return obj

        # 75% move, 25% appear/disappear.
        if self.rng.random() < 0.75:
            return obj.moved(self.rng)
        return obj.toggled()

    def generate(self, num_sessions: int = 100) -> GeneratedDataset:
        if num_sessions < 1:
            raise ValueError("num_sessions must be >= 1")

        # Reset RNG so generate() remains reproducible.
        self.rng = random.Random(self.seed)

        objects = self._make_static_objects() + self._make_dynamic_objects()
        state: Dict[str, CuboidObject] = {obj.object_id: obj for obj in objects}

        sessions: list[Session] = []
        changes: list[Change] = []
        object_history: list[Dict[str, CuboidObject]] = []

        occupied = self._all_voxels(state)
        sessions.append(Session(0, occupied))
        object_history.append(dict(state))

        for t in range(1, num_sessions):
            state = {oid: self._advance_object(obj) for oid, obj in state.items()}
            new_occupied = self._all_voxels(state)

            added = frozenset(new_occupied - occupied)
            removed = frozenset(occupied - new_occupied)

            changes.append(Change(t, added, removed))
            sessions.append(Session(t, new_occupied))
            object_history.append(dict(state))
            occupied = new_occupied

        dataset = GeneratedDataset(self.spec, sessions, changes, object_history)
        dataset.validate()
        return dataset
