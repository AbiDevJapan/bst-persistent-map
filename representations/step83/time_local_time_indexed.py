from __future__ import annotations
from .api import OutputMode


class TimeLocalTimeIndexedExplicitMap:
    """Step-8.5 time-local blocks with counted (bucket,session) postings."""
    name = 'time_local_tindex_explicit'
    native_kind = 'blocks'
    native_change_exact = False

    def __init__(self, core):
        self.core = core

    @property
    def last_query_stats(self): return self.core.last_query_stats
    @property
    def last_change_stats(self): return self.core.last_change_stats

    def region_query(self, region, t, *, output: OutputMode = 'voxels'):
        if output == 'native':
            return self.core.region_native(region, t, index_mode='time')
        return self.core.region_voxels(region, t, index_mode='time')

    def changes_query(self, region, ta, tb, *, output: OutputMode = 'voxels'):
        if output == 'native':
            return self.core.changes_native(region, ta, tb)
        return self.core.changes_voxels(region, ta, tb, index_mode='time')

    def compact_bytes(self):
        return self.core.common_storage_bytes(index_mode='time') + 16 * len(self.core.blocks)

    def stats(self):
        return self.core.stats(index_mode='time')
