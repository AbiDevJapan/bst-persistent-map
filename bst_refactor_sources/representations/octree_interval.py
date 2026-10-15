from __future__ import annotations

from dataclasses import dataclass, field

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds
from .interval_utils import (
    Signature,
    bits_required,
    build_occupied_intervals,
    prefix_active,
    signature_prefixes,
    tokens_by_depth,
)


@dataclass
class _IntervalNode:
    x0: int
    y0: int
    z0: int
    size: int
    children: dict[int, "_IntervalNode"] = field(default_factory=dict)
    # Used by OctreeIntervalMap (duplicated at each leaf).
    tokens: tuple[tuple[int, int], ...] = ()
    depth_index: dict[int, frozenset[int]] | None = None
    # Used by OctreeSharedSignatureMap.
    signature_id: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.size == 1


class _OctreeIntervalBase:
    """Common 3-D Octree traversal for the strengthened Step-6 baselines."""

    def __init__(self, dataset: GeneratedDataset):
        self.dataset = dataset
        self.time_bits = bits_required(len(dataset.sessions))
        max_dim = max(dataset.spec.nx, dataset.spec.ny, dataset.spec.nz)
        root_size = 1
        while root_size < max_dim:
            root_size *= 2
        self.root = _IntervalNode(0, 0, 0, root_size)
        self.node_count = 1
        self.leaf_count = 0
        self.last_query_stats = {"nodes_visited": 0, "leaves_tested": 0}
        self.last_change_stats = {"nodes_visited": 0, "leaves_tested": 0}

    def _child_index(self, node: _IntervalNode, v: Voxel) -> int:
        x, y, z = v
        half = node.size // 2
        return (
            (1 if x >= node.x0 + half else 0)
            | ((1 if y >= node.y0 + half else 0) << 1)
            | ((1 if z >= node.z0 + half else 0) << 2)
        )

    def _child_origin(self, node: _IntervalNode, idx: int):
        half = node.size // 2
        return (
            node.x0 + (half if (idx & 1) else 0),
            node.y0 + (half if (idx & 2) else 0),
            node.z0 + (half if (idx & 4) else 0),
            half,
        )

    def _leaf_for(self, v: Voxel, create: bool = True) -> _IntervalNode | None:
        node = self.root
        while not node.is_leaf:
            idx = self._child_index(node, v)
            child = node.children.get(idx)
            if child is None:
                if not create:
                    return None
                x0, y0, z0, size = self._child_origin(node, idx)
                child = _IntervalNode(x0, y0, z0, size)
                node.children[idx] = child
                self.node_count += 1
                if size == 1:
                    self.leaf_count += 1
            node = child
        return node

    @staticmethod
    def _intersects(node: _IntervalNode, bounds) -> bool:
        x0, x1, y0, y1, z0, z1 = bounds
        return not (
            node.x0 + node.size - 1 < x0 or node.x0 > x1
            or node.y0 + node.size - 1 < y0 or node.y0 > y1
            or node.z0 + node.size - 1 < z0 or node.z0 > z1
        )

    def _active(self, leaf: _IntervalNode, t: int) -> bool:
        raise NotImplementedError

    def _query(self, node, bounds, t, out, counters):
        counters["nodes_visited"] += 1
        if not self._intersects(node, bounds):
            return
        if node.is_leaf:
            counters["leaves_tested"] += 1
            if self._active(node, t):
                x0, x1, y0, y1, z0, z1 = bounds
                v = (node.x0, node.y0, node.z0)
                if x0 <= v[0] <= x1 and y0 <= v[1] <= y1 and z0 <= v[2] <= z1:
                    out.add(v)
            return
        for child in node.children.values():
            self._query(child, bounds, t, out, counters)

    def region_at_time(self, region: MetricBox, t: int) -> frozenset[Voxel]:
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        counters = {"nodes_visited": 0, "leaves_tested": 0}
        out: set[Voxel] = set()
        self._query(self.root, bounds, t, out, counters)
        self.last_query_stats = counters
        return frozenset(out)

    def snapshot(self, t: int) -> frozenset[Voxel]:
        s = self.dataset.spec
        return self.region_at_time((0, s.size_x_m, 0, s.size_y_m, 0, s.size_z_m), t)

    def _change_query(self, node, bounds, t_a, t_b, added, removed, counters):
        counters["nodes_visited"] += 1
        if not self._intersects(node, bounds):
            return
        if node.is_leaf:
            counters["leaves_tested"] += 1
            x0, x1, y0, y1, z0, z1 = bounds
            v = (node.x0, node.y0, node.z0)
            if not (x0 <= v[0] <= x1 and y0 <= v[1] <= y1 and z0 <= v[2] <= z1):
                return
            a = self._active(node, t_a)
            b = self._active(node, t_b)
            if a == b:
                return
            (added if b else removed).add(v)
            return
        for child in node.children.values():
            self._change_query(child, bounds, t_a, t_b, added, removed, counters)

    def changes(self, region: MetricBox, t_a: int, t_b: int):
        if t_b < t_a:
            t_a, t_b = t_b, t_a
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        counters = {"nodes_visited": 0, "leaves_tested": 0}
        added: set[Voxel] = set()
        removed: set[Voxel] = set()
        self._change_query(self.root, bounds, t_a, t_b, added, removed, counters)
        self.last_change_stats = counters
        return frozenset(added), frozenset(removed)


class OctreeIntervalMap(_OctreeIntervalBase):
    """
    Strong baseline C: Octree + exact temporal validity prefixes.

    Temporal prefixes are duplicated at each leaf. This isolates the gain from
    replacing transition histories with direct validity intervals while keeping
    a conventional spatial Octree.
    """

    name = "OctreeIntervalMap"

    def __init__(self, dataset: GeneratedDataset):
        super().__init__(dataset)
        self.temporal_prefixes = 0

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset):
        obj = cls(dataset)
        for voxel, signature in build_occupied_intervals(dataset).items():
            leaf = obj._leaf_for(voxel, create=True)
            assert leaf is not None
            tokens = signature_prefixes(signature, len(dataset.sessions))
            leaf.tokens = tokens
            leaf.depth_index = tokens_by_depth(tokens)
            obj.temporal_prefixes += len(tokens)
        return obj

    def _active(self, leaf: _IntervalNode, t: int) -> bool:
        return prefix_active(leaf.depth_index or {}, t, self.time_bits)

    def stats(self):
        return {
            "nodes": self.node_count,
            "leaves": self.leaf_count,
            "temporal_prefixes": self.temporal_prefixes,
            "time_bits": self.time_bits,
        }


class OctreeSharedSignatureMap(_OctreeIntervalBase):
    """
    Strong baseline D: Octree + validity prefixes + shared temporal signatures.

    Leaves reference a deduplicated signature table. This gives the Octree the
    same temporal-history sharing advantage used by the BST interval variants.
    """

    name = "OctreeSharedSignatureMap"

    def __init__(self, dataset: GeneratedDataset):
        super().__init__(dataset)
        self.signatures: list[Signature] = []
        self.signature_tokens: list[tuple[tuple[int, int], ...]] = []
        self.signature_depth_index: list[dict[int, frozenset[int]]] = []

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset):
        obj = cls(dataset)
        sig_to_id: dict[Signature, int] = {}
        for voxel, signature in build_occupied_intervals(dataset).items():
            sid = sig_to_id.get(signature)
            if sid is None:
                sid = len(obj.signatures)
                sig_to_id[signature] = sid
                obj.signatures.append(signature)
                tokens = signature_prefixes(signature, len(dataset.sessions))
                obj.signature_tokens.append(tokens)
                obj.signature_depth_index.append(tokens_by_depth(tokens))
            leaf = obj._leaf_for(voxel, create=True)
            assert leaf is not None
            leaf.signature_id = sid
        return obj

    def _active(self, leaf: _IntervalNode, t: int) -> bool:
        if leaf.signature_id < 0:
            return False
        return prefix_active(
            self.signature_depth_index[leaf.signature_id], t, self.time_bits
        )

    def stats(self):
        return {
            "nodes": self.node_count,
            "leaves": self.leaf_count,
            "unique_temporal_signatures": len(self.signatures),
            "temporal_prefixes": sum(len(x) for x in self.signature_tokens),
            "time_bits": self.time_bits,
        }
