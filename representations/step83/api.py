from __future__ import annotations

from typing import Protocol, Literal, runtime_checkable, Any

OutputMode = Literal['voxels','native']

@runtime_checkable
class LifelongMap(Protocol):
    """Common Step-8.3 map API.

    ``output='voxels'`` is the canonical equal-output path used for exact
    correctness and apples-to-apples timing.

    ``output='native'`` returns the representation's natural local-map
    primitive.  For adaptive maps this is clipped blocks; for voxel/Octree/
    snapshot/delta representations it remains voxel leaves/codes.
    """
    name: str
    native_kind: str

    def region_query(self, region, t: int, *, output: OutputMode='voxels') -> Any: ...
    def changes_query(self, region, ta: int, tb: int, *, output: OutputMode='voxels') -> Any: ...
    def compact_bytes(self) -> int: ...
    def stats(self) -> dict: ...
