from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel

Interval = tuple[int, int]  # [start, end)
Signature = tuple[Interval, ...]


def bits_required(n_values: int) -> int:
    if n_values <= 1:
        return 1
    return math.ceil(math.log2(n_values))


def build_occupied_intervals(dataset: GeneratedDataset) -> dict[Voxel, Signature]:
    """Convert transition history into exact occupied validity intervals."""
    starts: dict[Voxel, int] = {v: 0 for v in dataset.sessions[0].occupied}
    intervals: dict[Voxel, list[Interval]] = defaultdict(list)

    for change in dataset.changes:
        t = change.session_id
        for v in change.removed:
            start = starts.pop(v, None)
            if start is None:
                raise AssertionError(f'removed inactive voxel {v} at t={t}')
            if start < t:
                intervals[v].append((start, t))
        for v in change.added:
            if v in starts:
                raise AssertionError(f'added already-active voxel {v} at t={t}')
            starts[v] = t

    end = len(dataset.sessions)
    for v, start in starts.items():
        if start < end:
            intervals[v].append((start, end))

    return {v: tuple(xs) for v, xs in intervals.items() if xs}


def dyadic_cover(start: int, end: int, time_bits: int) -> tuple[tuple[int, int], ...]:
    """
    Canonical exact cover of [start,end) by aligned dyadic blocks.

    Returns (prefix_len, prefix_value) over a `time_bits` binary time axis.
    A block of size 2^(time_bits-prefix_len) is one temporal BST prefix.
    """
    if not (0 <= start <= end <= (1 << time_bits)):
        raise ValueError((start, end, time_bits))
    out: list[tuple[int, int]] = []
    cur = start
    while cur < end:
        # Largest aligned power-of-two block starting at cur.
        if cur == 0:
            size = 1 << time_bits
        else:
            size = cur & -cur
        remaining = end - cur
        while size > remaining:
            size >>= 1
        exponent = size.bit_length() - 1
        prefix_len = time_bits - exponent
        prefix_value = cur >> exponent
        out.append((prefix_len, prefix_value))
        cur += size
    return tuple(out)


def signature_prefixes(signature: Signature, num_sessions: int) -> tuple[tuple[int, int], ...]:
    tb = bits_required(num_sessions)
    tokens: list[tuple[int, int]] = []
    for start, end in signature:
        tokens.extend(dyadic_cover(start, end, tb))
    return tuple(tokens)


def prefix_active(tokens_by_depth: dict[int, frozenset[int]], t: int, time_bits: int) -> bool:
    """Membership query using temporal prefix ancestry, not predecessor search."""
    for depth, prefixes in tokens_by_depth.items():
        if depth == 0:
            prefix = 0
        else:
            prefix = t >> (time_bits - depth)
        if prefix in prefixes:
            return True
    return False


def tokens_by_depth(tokens: tuple[tuple[int, int], ...]) -> dict[int, frozenset[int]]:
    d: dict[int, set[int]] = defaultdict(set)
    for depth, prefix in tokens:
        d[depth].add(prefix)
    return {depth: frozenset(vals) for depth, vals in d.items()}
