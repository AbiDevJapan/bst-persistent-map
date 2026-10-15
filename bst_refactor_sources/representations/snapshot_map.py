from __future__ import annotations

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds, voxel_in_bounds


class SnapshotMap:
    """
    Baseline A: store a complete occupancy snapshot for every session.
    """

    name = "SnapshotMap"

    def __init__(self, dataset: GeneratedDataset):
        self.dataset = dataset
        self._sessions: dict[int, frozenset[Voxel]] = {}
        self.last_query_stats = {"records_scanned": 0}
        self.last_change_stats = {"records_scanned": 0}

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset) -> "SnapshotMap":
        obj = cls(dataset)
        for session in dataset.sessions:
            obj.insert_session(session.session_id, session.occupied)
        return obj

    def insert_session(self, t: int, occupied: frozenset[Voxel]) -> None:
        self._sessions[t] = frozenset(occupied)

    def snapshot(self, t: int) -> frozenset[Voxel]:
        return self._sessions[t]

    def region_at_time(self, region: MetricBox, t: int) -> frozenset[Voxel]:
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        records = self._sessions[t]
        self.last_query_stats = {"records_scanned": len(records)}
        return frozenset(v for v in records if voxel_in_bounds(v, bounds))

    def changes(
        self,
        region: MetricBox,
        t_a: int,
        t_b: int,
    ) -> tuple[frozenset[Voxel], frozenset[Voxel]]:
        if t_b < t_a:
            t_a, t_b = t_b, t_a
        a = self.region_at_time(region, t_a)
        sa = self.last_query_stats["records_scanned"]
        b = self.region_at_time(region, t_b)
        sb = self.last_query_stats["records_scanned"]
        self.last_change_stats = {"records_scanned": sa + sb}
        return frozenset(b - a), frozenset(a - b)

    def stats(self) -> dict:
        return {
            "sessions": len(self._sessions),
            "stored_voxel_records": sum(len(v) for v in self._sessions.values()),
        }
