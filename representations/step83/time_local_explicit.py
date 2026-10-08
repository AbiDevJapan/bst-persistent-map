from __future__ import annotations
from .api import OutputMode


class TimeLocalExplicitMap:
    name = 'time_local_explicit'
    native_kind = 'blocks'
    native_change_exact = False

    def __init__(self, core):
        self.core = core

    @property
    def last_query_stats(self): return self.core.last_query_stats
    @property
    def last_change_stats(self): return self.core.last_change_stats

    def region_query(self, region, t, *, output: OutputMode = 'voxels'):
        return self.core.region_native(region, t) if output == 'native' else self.core.region_voxels(region, t)

    def changes_query(self, region, ta, tb, *, output: OutputMode = 'voxels'):
        if output == 'native':
            return self.core.changes_native(region, ta, tb)
        return self.core.changes_voxels(region, ta, tb)

    def compact_bytes(self):
        # Same logical record format as AdaptiveExplicit: uint16 x0,y0,z0,sx,sy,sz
        # plus uint32 shared temporal-signature id.
        return self.core.common_storage_bytes() + 16 * len(self.core.blocks)

    def stats(self): return self.core.stats()
