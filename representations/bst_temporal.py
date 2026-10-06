from __future__ import annotations

from dataclasses import dataclass

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .bst_encoder import LocalMetricBSTEncoder, PrefixBounds
from .common import MetricBox, metric_box_to_voxel_bounds


@dataclass(frozen=True)
class _BSTEvent:
    key: int
    x: int
    y: int
    z: int
    t: int
    state: bool


class BSTTemporalMapPrototype:
    """
    Step-3 BST-style 4-D event map.

    Each occupancy transition is addressed by a hierarchical binary key over
    x/y/z/t.  Events are grouped by a leading key prefix.  Region/time queries
    first reject non-intersecting prefix buckets, then inspect exact events.

    This version intentionally uses FIXED coordinate/time precision.  Adaptive
    prefix depth belongs to a later step.
    """

    name = "BSTTemporalMapPrototype"

    def __init__(
        self,
        dataset: GeneratedDataset,
        bucket_prefix_bits: int = 10,
    ):
        self.dataset = dataset
        self.encoder = LocalMetricBSTEncoder(
            dataset.spec.nx,
            dataset.spec.ny,
            dataset.spec.nz,
            len(dataset.sessions),
        )
        self.bucket_prefix_bits = min(
            max(1, bucket_prefix_bits),
            self.encoder.total_bits,
        )

        self._buckets: dict[int, list[_BSTEvent]] = {}
        self._bucket_bounds: dict[int, PrefixBounds] = {}
        self.event_count = 0

        self.last_query_stats = {
            "buckets_examined": 0,
            "buckets_selected": 0,
            "events_scanned": 0,
        }
        self.last_change_stats = {
            "buckets_examined": 0,
            "buckets_selected": 0,
            "events_scanned": 0,
        }

    @classmethod
    def from_dataset(
        cls,
        dataset: GeneratedDataset,
        bucket_prefix_bits: int = 10,
    ) -> "BSTTemporalMapPrototype":
        obj = cls(dataset, bucket_prefix_bits)

        for x, y, z in dataset.sessions[0].occupied:
            obj._insert_event((x, y, z), 0, True)

        for change in dataset.changes:
            for v in change.removed:
                obj._insert_event(v, change.session_id, False)
            for v in change.added:
                obj._insert_event(v, change.session_id, True)

        # Chronological order makes "latest state" reduction deterministic.
        for events in obj._buckets.values():
            events.sort(key=lambda e: e.t)

        return obj

    def _insert_event(self, v: Voxel, t: int, state: bool) -> None:
        x, y, z = v
        key = self.encoder.encode(x, y, z, t)
        prefix = self.encoder.prefix(key, self.bucket_prefix_bits)

        if prefix not in self._buckets:
            self._buckets[prefix] = []
            self._bucket_bounds[prefix] = self.encoder.prefix_bounds(
                prefix,
                self.bucket_prefix_bits,
            )

        self._buckets[prefix].append(
            _BSTEvent(key, x, y, z, t, state)
        )
        self.event_count += 1

    @staticmethod
    def _spatial_intersects(
        pb: PrefixBounds,
        bounds: tuple[int, int, int, int, int, int],
    ) -> bool:
        x0, x1, y0, y1, z0, z1 = bounds
        return not (
            pb.x[1] < x0 or pb.x[0] > x1
            or pb.y[1] < y0 or pb.y[0] > y1
            or pb.z[1] < z0 or pb.z[0] > z1
        )

    def region_at_time(self, region: MetricBox, t: int) -> frozenset[Voxel]:
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        x0, x1, y0, y1, z0, z1 = bounds

        latest: dict[Voxel, tuple[int, bool]] = {}
        stats = {
            "buckets_examined": 0,
            "buckets_selected": 0,
            "events_scanned": 0,
        }

        for prefix, events in self._buckets.items():
            stats["buckets_examined"] += 1
            pb = self._bucket_bounds[prefix]

            if pb.t[0] > t:
                continue
            if not self._spatial_intersects(pb, bounds):
                continue

            stats["buckets_selected"] += 1

            for event in events:
                stats["events_scanned"] += 1
                if event.t > t:
                    continue
                if not (
                    x0 <= event.x <= x1
                    and y0 <= event.y <= y1
                    and z0 <= event.z <= z1
                ):
                    continue

                v = (event.x, event.y, event.z)
                prev = latest.get(v)
                if prev is None or event.t >= prev[0]:
                    latest[v] = (event.t, event.state)

        self.last_query_stats = stats
        return frozenset(v for v, (_, state) in latest.items() if state)

    def snapshot(self, t: int) -> frozenset[Voxel]:
        s = self.dataset.spec
        return self.region_at_time(
            (0.0, s.size_x_m, 0.0, s.size_y_m, 0.0, s.size_z_m),
            t,
        )

    def changes(
        self,
        region: MetricBox,
        t_a: int,
        t_b: int,
    ) -> tuple[frozenset[Voxel], frozenset[Voxel]]:
        if t_b < t_a:
            t_a, t_b = t_b, t_a
        a = self.region_at_time(region, t_a)
        sa = dict(self.last_query_stats)
        b = self.region_at_time(region, t_b)
        sb = dict(self.last_query_stats)
        self.last_change_stats = {
            k: sa.get(k, 0) + sb.get(k, 0)
            for k in set(sa) | set(sb)
        }
        return frozenset(b - a), frozenset(a - b)

    def stats(self) -> dict:
        return {
            "events": self.event_count,
            "prefix_buckets": len(self._buckets),
            "bucket_prefix_bits": self.bucket_prefix_bits,
            "total_key_bits": self.encoder.total_bits,
            "dimension_bits": dict(self.encoder.widths),
        }
