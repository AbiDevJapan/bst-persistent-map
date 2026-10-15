"""Prefix indexes for BST-ID region matching.

This module provides an indexed alternative to the linear scans used by
``bst_id.region_algebra``. It is intentionally kept separate from the
reference algebra so that the reference implementation remains simple and
transparent.

The index is built independently for each active BST-ID axis. Each axis uses
a binary prefix trie. Trie nodes store:

* ``terminals``: cells whose prefix ends exactly at that node.
* ``members``: cells whose prefix passes through that node (including
  terminals below it).

Multi-axis queries retrieve candidate sets independently on x, y, f, and t and
intersect those sets. Therefore a one-axis prefix lookup is O(L + output),
where L <= 32, while a complete multi-axis query also includes posting-set
intersection cost. This module does not claim that a four-axis query is
strictly O(L) independent of candidate/output size.

The package currently uses axis name ``f`` for altitude/height, matching
``region_algebra.AXES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

from .region_algebra import AXES, Cell, meet, normalize


@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    terminals: Set[Cell] = field(default_factory=set)
    members: Set[Cell] = field(default_factory=set)


class _AxisPrefixTrie:
    """Binary prefix trie for one BST-ID axis."""

    def __init__(self, axis: int):
        self.axis = axis
        self.root = _TrieNode()

    def insert(self, cell: Cell) -> None:
        z = cell.zooms[self.axis]
        value = cell.values[self.axis]

        node = self.root
        node.members.add(cell)

        for shift in range(z - 1, -1, -1):
            bit = (value >> shift) & 1
            child = node.children.get(bit)
            if child is None:
                child = _TrieNode()
                node.children[bit] = child
            node = child
            node.members.add(cell)

        node.terminals.add(cell)

    def discard(self, cell: Cell) -> None:
        """Remove a cell if present and prune empty trie nodes."""
        z = cell.zooms[self.axis]
        value = cell.values[self.axis]

        node = self.root
        nodes: List[_TrieNode] = [node]
        bits: List[int] = []

        for shift in range(z - 1, -1, -1):
            bit = (value >> shift) & 1
            child = node.children.get(bit)
            if child is None:
                return
            bits.append(bit)
            node = child
            nodes.append(node)

        node.terminals.discard(cell)
        for path_node in nodes:
            path_node.members.discard(cell)

        # Prune empty nodes from leaf toward root.
        for depth in range(len(bits) - 1, -1, -1):
            parent = nodes[depth]
            bit = bits[depth]
            child = parent.children.get(bit)
            if child is None:
                continue
            if child.members or child.terminals or child.children:
                break
            del parent.children[bit]

    def ancestor_matches(self, zoom: int, value: int) -> Set[Cell]:
        """Cells whose stored prefix is an ancestor of/equal to the query."""
        out: Set[Cell] = set()
        node = self.root

        for shift in range(zoom - 1, -1, -1):
            bit = (value >> shift) & 1
            node = node.children.get(bit)
            if node is None:
                break
            out.update(node.terminals)

        return out

    def descendant_matches(self, zoom: int, value: int) -> Set[Cell]:
        """Cells whose stored prefix is a descendant of/equal to the query."""
        node = self.root

        for shift in range(zoom - 1, -1, -1):
            bit = (value >> shift) & 1
            node = node.children.get(bit)
            if node is None:
                return set()

        return set(node.members)

    def compatible_matches(self, zoom: int, value: int) -> Set[Cell]:
        """Cells whose stored prefix is prefix-compatible with the query."""
        out: Set[Cell] = set()
        node = self.root

        # Stored prefixes ending on the query path are ancestors.
        for shift in range(zoom - 1, -1, -1):
            bit = (value >> shift) & 1
            node = node.children.get(bit)
            if node is None:
                return out
            out.update(node.terminals)

        # All stored prefixes below the query node are descendants.
        out.update(node.members)
        return out

    def clear(self) -> None:
        self.root = _TrieNode()


def _intersect_postings(postings: List[Set[Cell]]) -> Set[Cell]:
    """Intersect postings, starting from the smallest set."""
    if not postings:
        return set()

    postings.sort(key=len)
    out = set(postings[0])
    for posting in postings[1:]:
        out.intersection_update(posting)
        if not out:
            break
    return out


class RegionPrefixIndex:
    """Multi-axis prefix index for one BST-ID presence domain.

    The index stores the same ``Cell`` objects in one binary trie per active
    axis. A multi-axis query obtains an axis-specific candidate set and
    intersects the postings.

    This is intended for repeated queries over an already-built region. For a
    one-shot query, index construction cost may outweigh the benefit.
    """

    def __init__(self, cells: Iterable[Cell] = ()):
        self._flags: Optional[int] = None
        self._cells: Set[Cell] = set()
        self._tries: Tuple[_AxisPrefixTrie, ...] = tuple(
            _AxisPrefixTrie(i) for i in range(len(AXES))
        )
        self.update(cells)

    @property
    def flags(self) -> Optional[int]:
        return self._flags

    @property
    def cells(self) -> FrozenSet[Cell]:
        return frozenset(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def __bool__(self) -> bool:
        return bool(self._cells)

    def _active_axes(self) -> Tuple[int, ...]:
        if self._flags is None:
            return tuple()
        return tuple(
            i
            for i in range(len(AXES))
            if (self._flags >> (len(AXES) - 1 - i)) & 1
        )

    def _check_domain(self, cell: Cell) -> None:
        if self._flags is not None and cell.flags != self._flags:
            raise ValueError(
                "RegionPrefixIndex requires the same presence vector "
                "for indexed and query cells"
            )

    def add(self, cell: Cell) -> None:
        """Insert one cell. Duplicate cells are ignored."""
        if self._flags is None:
            self._flags = cell.flags
        self._check_domain(cell)

        if cell in self._cells:
            return

        self._cells.add(cell)
        for axis in self._active_axes():
            self._tries[axis].insert(cell)

    def update(self, cells: Iterable[Cell]) -> None:
        for cell in cells:
            self.add(cell)

    def discard(self, cell: Cell) -> None:
        """Remove one cell if present."""
        if self._flags is None:
            return
        self._check_domain(cell)

        if cell not in self._cells:
            return

        for axis in self._active_axes():
            self._tries[axis].discard(cell)
        self._cells.remove(cell)

        if not self._cells:
            self._flags = None

    def remove(self, cell: Cell) -> None:
        """Remove one cell, raising KeyError if it is absent."""
        if cell not in self._cells:
            raise KeyError(cell)
        self.discard(cell)

    def clear(self) -> None:
        self._cells.clear()
        self._flags = None
        for trie in self._tries:
            trie.clear()

    def subsuming_cells(self, query: Cell) -> FrozenSet[Cell]:
        """Return indexed cells r satisfying C(query) subseteq C(r)."""
        if not self._cells:
            return frozenset()
        self._check_domain(query)

        postings: List[Set[Cell]] = []
        for axis in self._active_axes():
            postings.append(
                self._tries[axis].ancestor_matches(
                    query.zooms[axis], query.values[axis]
                )
            )
        return frozenset(_intersect_postings(postings))

    def subsumed_cells(self, container: Cell) -> FrozenSet[Cell]:
        """Return indexed cells r satisfying C(r) subseteq C(container)."""
        if not self._cells:
            return frozenset()
        self._check_domain(container)

        postings: List[Set[Cell]] = []
        for axis in self._active_axes():
            postings.append(
                self._tries[axis].descendant_matches(
                    container.zooms[axis], container.values[axis]
                )
            )
        return frozenset(_intersect_postings(postings))

    def compatible_candidates(self, query: Cell) -> FrozenSet[Cell]:
        """Return indexed cells prefix-compatible with query on every axis."""
        if not self._cells:
            return frozenset()
        self._check_domain(query)

        postings: List[Set[Cell]] = []
        for axis in self._active_axes():
            postings.append(
                self._tries[axis].compatible_matches(
                    query.zooms[axis], query.values[axis]
                )
            )
        return frozenset(_intersect_postings(postings))

    def contains(self, query: Cell) -> bool:
        """True iff some indexed prefix subsumes the query cell."""
        return bool(self.subsuming_cells(query))

    def iter_compatible_pairs(
        self, queries: Iterable[Cell]
    ) -> Iterator[Tuple[Cell, Cell]]:
        """Yield (indexed_cell, query_cell) compatible pairs."""
        for query in queries:
            self._check_domain(query)
            for candidate in self.compatible_candidates(query):
                yield candidate, query

    def count_compatible_pairs(self, queries: Iterable[Cell]) -> int:
        """Count candidate pairs produced by indexed compatibility lookup."""
        count = 0
        for query in queries:
            self._check_domain(query)
            count += len(self.compatible_candidates(query))
        return count

    def intersection(self, queries: Iterable[Cell]) -> Tuple[Cell, ...]:
        """Indexed region intersection followed by reference normalization."""
        out: Set[Cell] = set()

        for candidate, query in self.iter_compatible_pairs(queries):
            cell = meet(candidate, query)
            if cell is not None:
                out.add(cell)

        return normalize(out)


def intersection_indexed(
    a: Iterable[Cell], b: Iterable[Cell]
) -> Tuple[Cell, ...]:
    """One-shot indexed intersection.

    The smaller input is indexed to limit index memory. For repeated queries,
    construct ``RegionPrefixIndex`` once and call ``index.intersection(...)``
    instead.
    """
    aa = tuple(a)
    bb = tuple(b)

    if not aa or not bb:
        return tuple()

    if len(aa) <= len(bb):
        return RegionPrefixIndex(aa).intersection(bb)
    return RegionPrefixIndex(bb).intersection(aa)


# ---------------------------------------------------------------------------
# Trie-indexed normalization
# ---------------------------------------------------------------------------

@dataclass
class NormalizeIndexStats:
    """Instrumentation for trie-indexed normalization."""

    passes: int = 0
    subsumption_queries: int = 0
    subsumed_removed: int = 0
    index_insertions: int = 0
    sibling_merges: int = 0
    max_region_size: int = 0


def _normalization_sort_key(c: Cell) -> Tuple:
    """Same coarse-first ordering used by region_algebra.normalize()."""
    return (sum(c.zooms), c.zooms, c.values)


def _indexed_subsumption_filter(
    region: Set[Cell],
    stats: Optional[NormalizeIndexStats] = None,
) -> Tuple[Set[Cell], bool]:
    """N1 using an incrementally built multi-axis prefix index.

    Cells are processed in the same coarse-first order as the reference
    implementation. Previously retained cells are indexed. Therefore a query
    only needs to determine whether one retained cell subsumes the current
    cell; a full scan over ``keep`` is avoided.

    The per-axis trie walk is bounded by L <= 32. Multi-axis retrieval still
    includes posting-set intersection/output cost, so this function should not
    be described as strict O(L) independent of candidate size.
    """
    ordered = sorted(region, key=_normalization_sort_key)
    keep: List[Cell] = []
    index = RegionPrefixIndex()
    changed = False

    for cell in ordered:
        if stats is not None:
            stats.subsumption_queries += 1

        if index.contains(cell):
            changed = True
            if stats is not None:
                stats.subsumed_removed += 1
            continue

        keep.append(cell)
        index.add(cell)
        if stats is not None:
            stats.index_insertions += 1

    return set(keep), changed


def _merge_siblings_once_indexed(region: Set[Cell], axis: int) -> bool:
    """One sibling merge using the same policy as the reference normalizer.

    This deliberately keeps the one-merge-then-restart behavior so that
    ``normalize_indexed`` is an apples-to-apples indexed N1 alternative to the
    current reference implementation.
    """
    buckets: Dict[Tuple, Dict[int, Cell]] = {}

    for c in region:
        if not c.active(axis) or c.zooms[axis] <= 1:
            continue

        z, v = c.zooms[axis], c.values[axis]
        other = tuple(
            (c.zooms[j], c.values[j]) if j != axis else (z - 1, v >> 1)
            for j in range(4)
        )
        key = (c.flags, axis, other)
        buckets.setdefault(key, {})[v & 1] = c

    for bits in buckets.values():
        if 0 not in bits or 1 not in bits:
            continue

        c0, c1 = bits[0], bits[1]
        if c0.values[axis] >> 1 != c1.values[axis] >> 1:
            continue

        parent = c0.replace_axis(
            axis,
            c0.zooms[axis] - 1,
            c0.values[axis] >> 1,
        )
        region.remove(c0)
        region.remove(c1)
        region.add(parent)
        return True

    return False


def _merge_siblings_batch(region: Set[Cell], axis: int) -> int:
    """Merge all currently available disjoint sibling pairs on one axis.

    This is an optimized variant of N2. It preserves the fixed axis order but
    removes all sibling pairs available at the selected axis before restarting
    N1. Newly created parent-level sibling relations are handled on the next
    pass.

    The validation script compares the final output exactly against the current
    reference normalizer before this variant should be used in reported results.
    """
    buckets: Dict[Tuple, Dict[int, Cell]] = {}

    for c in region:
        if not c.active(axis) or c.zooms[axis] <= 1:
            continue

        z, v = c.zooms[axis], c.values[axis]
        other = tuple(
            (c.zooms[j], c.values[j]) if j != axis else (z - 1, v >> 1)
            for j in range(4)
        )
        key = (c.flags, axis, other)
        buckets.setdefault(key, {})[v & 1] = c

    merges: List[Tuple[Cell, Cell, Cell]] = []

    for bits in buckets.values():
        if 0 not in bits or 1 not in bits:
            continue

        c0, c1 = bits[0], bits[1]
        if c0.values[axis] >> 1 != c1.values[axis] >> 1:
            continue

        parent = c0.replace_axis(
            axis,
            c0.zooms[axis] - 1,
            c0.values[axis] >> 1,
        )
        merges.append((c0, c1, parent))

    for c0, c1, parent in merges:
        # Pairs are disjoint by construction, but use discard defensively.
        region.discard(c0)
        region.discard(c1)
        region.add(parent)

    return len(merges)


def _normalize_indexed_impl(
    cells: Iterable[Cell],
    *,
    batch_siblings: bool,
) -> Tuple[Tuple[Cell, ...], NormalizeIndexStats]:
    """Shared implementation for the indexed normalizers."""
    region = set(cells)
    stats = NormalizeIndexStats(max_region_size=len(region))

    if not region:
        return tuple(), stats

    flags = next(iter(region)).flags
    if any(c.flags != flags for c in region):
        raise ValueError("all cells must share a presence vector")

    changed = True
    while changed:
        stats.passes += 1
        stats.max_region_size = max(stats.max_region_size, len(region))
        changed = False

        # N1: trie-indexed subsumption removal.
        region, removed = _indexed_subsumption_filter(region, stats)
        changed = changed or removed

        # N2: same fixed axis order x,y,f,t as the reference implementation.
        for axis in range(4):
            if batch_siblings:
                merged = _merge_siblings_batch(region, axis)
                if merged:
                    stats.sibling_merges += merged
                    changed = True
                    break
            else:
                if _merge_siblings_once_indexed(region, axis):
                    stats.sibling_merges += 1
                    changed = True
                    break

    return tuple(sorted(region)), stats


def normalize_indexed(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    """Normalize with trie-indexed N1 and reference-compatible one-pair N2.

    This is the primary apples-to-apples comparison against
    ``region_algebra.normalize``. Only the subsumption-removal stage is changed
    from a linear scan to prefix-indexed lookup; sibling merging keeps the
    current one-pair-then-restart policy.
    """
    out, _ = _normalize_indexed_impl(cells, batch_siblings=False)
    return out


def normalize_indexed_profile(
    cells: Iterable[Cell],
) -> Tuple[Tuple[Cell, ...], NormalizeIndexStats]:
    """Return indexed-normalization output plus instrumentation."""
    return _normalize_indexed_impl(cells, batch_siblings=False)


def normalize_indexed_batch(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    """Experimental indexed normalizer with batch sibling merging.

    Use only after exact-output validation against the current reference
    normalizer for the workloads being reported.
    """
    out, _ = _normalize_indexed_impl(cells, batch_siblings=True)
    return out


def normalize_indexed_batch_profile(
    cells: Iterable[Cell],
) -> Tuple[Tuple[Cell, ...], NormalizeIndexStats]:
    """Return batch-indexed normalization output plus instrumentation."""
    return _normalize_indexed_impl(cells, batch_siblings=True)
