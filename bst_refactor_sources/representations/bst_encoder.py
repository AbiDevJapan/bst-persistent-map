from __future__ import annotations

from dataclasses import dataclass
import math


DIMS = ("x", "y", "z", "t")


def bits_required(n_values: int) -> int:
    if n_values <= 1:
        return 1
    return math.ceil(math.log2(n_values))


@dataclass(frozen=True)
class PrefixBounds:
    x: tuple[int, int]
    y: tuple[int, int]
    z: tuple[int, int]
    t: tuple[int, int]


class LocalMetricBSTEncoder:
    """
    Step-3 *prototype* encoder for local robotic coordinates.

    It is intentionally isolated behind this class so that the exact upstream
    BST-ID encoder can later replace it without changing the map API.

    The encoder quantizes x/y/z to voxel indices and t to an integer session,
    then interleaves significant bits from x, y, z, t in round-robin order.
    Therefore every leading bit prefix identifies a hierarchical 4-D
    spatiotemporal region.

    Do NOT cite benchmark results from this class as results of the exact
    upstream BST-ID specification until that encoder has been integrated and
    cross-validated.
    """

    def __init__(self, nx: int, ny: int, nz: int, num_sessions: int):
        self.widths = {
            "x": bits_required(nx),
            "y": bits_required(ny),
            "z": bits_required(nz),
            "t": bits_required(num_sessions),
        }
        self.limits = {
            "x": nx,
            "y": ny,
            "z": nz,
            "t": num_sessions,
        }

        # Round-robin over each dimension's bits, always using that
        # dimension's next most-significant bit.
        schedule: list[tuple[str, int]] = []
        max_width = max(self.widths.values())
        for round_idx in range(max_width):
            for dim in DIMS:
                width = self.widths[dim]
                if round_idx < width:
                    bit_index = width - 1 - round_idx
                    schedule.append((dim, bit_index))

        self.schedule = tuple(schedule)
        self.total_bits = len(self.schedule)

    def encode(self, x: int, y: int, z: int, t: int) -> int:
        values = {"x": x, "y": y, "z": z, "t": t}
        for dim, value in values.items():
            if not (0 <= value < self.limits[dim]):
                raise ValueError(f"{dim}={value} outside [0,{self.limits[dim]-1}]")

        key = 0
        for dim, bit_index in self.schedule:
            key = (key << 1) | ((values[dim] >> bit_index) & 1)
        return key

    def decode(self, key: int) -> tuple[int, int, int, int]:
        values = {d: 0 for d in DIMS}
        for i, (dim, bit_index) in enumerate(self.schedule):
            shift = self.total_bits - 1 - i
            bit = (key >> shift) & 1
            values[dim] |= bit << bit_index
        return values["x"], values["y"], values["z"], values["t"]

    def bitstring(self, key: int) -> str:
        return format(key, f"0{self.total_bits}b")

    def prefix(self, key: int, prefix_len: int) -> int:
        if not (0 <= prefix_len <= self.total_bits):
            raise ValueError("invalid prefix length")
        if prefix_len == 0:
            return 0
        return key >> (self.total_bits - prefix_len)

    def prefix_bounds(self, prefix_value: int, prefix_len: int) -> PrefixBounds:
        if not (0 <= prefix_len <= self.total_bits):
            raise ValueError("invalid prefix length")

        known_values = {d: 0 for d in DIMS}
        known_counts = {d: 0 for d in DIMS}

        for i in range(prefix_len):
            dim, bit_index = self.schedule[i]
            bit = (prefix_value >> (prefix_len - 1 - i)) & 1
            known_values[dim] |= bit << bit_index
            known_counts[dim] += 1

        bounds = {}
        for dim in DIMS:
            width = self.widths[dim]
            count = known_counts[dim]
            remaining = width - count

            # Bits for a dimension are consumed from MSB to LSB, so the
            # unknown suffix occupies the low-order 'remaining' bits.
            low = known_values[dim]
            high = low | ((1 << remaining) - 1 if remaining else 0)

            # Clamp unused power-of-two address space.
            high = min(high, self.limits[dim] - 1)
            bounds[dim] = (low, high)

        return PrefixBounds(
            x=bounds["x"],
            y=bounds["y"],
            z=bounds["z"],
            t=bounds["t"],
        )
