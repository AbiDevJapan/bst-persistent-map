from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds


@dataclass
class _OctreeNode:
    x0: int
    y0: int
    z0: int
    size: int
    children: dict[int, "_OctreeNode"] = field(default_factory=dict)
    times: list[int] = field(default_factory=list)
    states: list[bool] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.size == 1

    def state_at(self, t: int) -> bool:
        if not self.times:
            return False
        idx = bisect_right(self.times, t) - 1
        if idx < 0:
            return False
        return self.states[idx]


class OctreeTemporalMap:
    """
    Baseline B: 3-D Octree spatial index with transition history on each leaf.

    This is deliberately a strong baseline: unchanged voxels do not duplicate
    their state every session.
    """

    name = "OctreeTemporalMap"

    def __init__(self, dataset: GeneratedDataset):
        self.dataset = dataset
        max_dim = max(dataset.spec.nx, dataset.spec.ny, dataset.spec.nz)
        root_size = 1
        while root_size < max_dim:
            root_size *= 2

        self.root = _OctreeNode(0, 0, 0, root_size)
        self.node_count = 1
        self.leaf_count = 0
        self.history_entries = 0
        self.last_query_nodes_visited = 0
        self.last_query_stats = {"nodes_visited": 0}
        self.last_change_stats = {"nodes_visited": 0}

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset) -> "OctreeTemporalMap":
        obj = cls(dataset)

        # Initial occupied state.
        for v in dataset.sessions[0].occupied:
            obj._set_state(v, 0, True)

        # Only store state transitions thereafter.
        for change in dataset.changes:
            for v in change.removed:
                obj._set_state(v, change.session_id, False)
            for v in change.added:
                obj._set_state(v, change.session_id, True)

        return obj

    def _child_index(self, node: _OctreeNode, v: Voxel) -> int:
        x, y, z = v
        half = node.size // 2
        ix = 1 if x >= node.x0 + half else 0
        iy = 1 if y >= node.y0 + half else 0
        iz = 1 if z >= node.z0 + half else 0
        return ix | (iy << 1) | (iz << 2)

    def _child_origin(self, node: _OctreeNode, idx: int):
        half = node.size // 2
        return (
            node.x0 + (half if (idx & 1) else 0),
            node.y0 + (half if (idx & 2) else 0),
            node.z0 + (half if (idx & 4) else 0),
            half,
        )

    def _leaf_for(self, v: Voxel, create: bool) -> _OctreeNode | None:
        node = self.root
        while not node.is_leaf:
            idx = self._child_index(node, v)
            child = node.children.get(idx)
            if child is None:
                if not create:
                    return None
                x0, y0, z0, size = self._child_origin(node, idx)
                child = _OctreeNode(x0, y0, z0, size)
                node.children[idx] = child
                self.node_count += 1
                if size == 1:
                    self.leaf_count += 1
            node = child
        return node

    def _set_state(self, v: Voxel, t: int, state: bool) -> None:
        leaf = self._leaf_for(v, create=True)
        assert leaf is not None

        if leaf.times and leaf.times[-1] == t:
            leaf.states[-1] = state
            return

        # Avoid redundant transitions.
        if leaf.states and leaf.states[-1] == state:
            return

        leaf.times.append(t)
        leaf.states.append(state)
        self.history_entries += 1

    @staticmethod
    def _intersects(
        node: _OctreeNode,
        bounds: tuple[int, int, int, int, int, int],
    ) -> bool:
        x0, x1, y0, y1, z0, z1 = bounds
        nx1 = node.x0 + node.size - 1
        ny1 = node.y0 + node.size - 1
        nz1 = node.z0 + node.size - 1
        return not (
            nx1 < x0 or node.x0 > x1
            or ny1 < y0 or node.y0 > y1
            or nz1 < z0 or node.z0 > z1
        )

    def _query_node(
        self,
        node: _OctreeNode,
        bounds: tuple[int, int, int, int, int, int],
        t: int,
        out: set[Voxel],
    ) -> None:
        self.last_query_nodes_visited += 1
        if not self._intersects(node, bounds):
            return

        if node.is_leaf:
            if node.state_at(t):
                v = (node.x0, node.y0, node.z0)
                x0, x1, y0, y1, z0, z1 = bounds
                x, y, z = v
                if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                    out.add(v)
            return

        for child in node.children.values():
            self._query_node(child, bounds, t, out)

    def region_at_time(self, region: MetricBox, t: int) -> frozenset[Voxel]:
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        self.last_query_nodes_visited = 0
        self.last_query_stats = {"nodes_visited": 0}
        self.last_change_stats = {"nodes_visited": 0}
        out: set[Voxel] = set()
        self._query_node(self.root, bounds, t, out)
        self.last_query_stats = {"nodes_visited": self.last_query_nodes_visited}
        return frozenset(out)

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
        va = self.last_query_stats["nodes_visited"]
        b = self.region_at_time(region, t_b)
        vb = self.last_query_stats["nodes_visited"]
        self.last_change_stats = {"nodes_visited": va + vb}
        return frozenset(b - a), frozenset(a - b)

    def stats(self) -> dict:
        return {
            "nodes": self.node_count,
            "leaves": self.leaf_count,
            "history_entries": self.history_entries,
        }
