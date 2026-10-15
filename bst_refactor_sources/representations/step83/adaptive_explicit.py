from __future__ import annotations
from .api import OutputMode

class ExplicitAdaptiveMap:
    name='adaptive_explicit'
    native_kind='blocks'

    def __init__(self, core):
        self.core=core

    @property
    def last_query_stats(self): return self.core.last_query_stats
    @property
    def last_change_stats(self): return self.core.last_change_stats

    def region_query(self,region,t,*,output:OutputMode='voxels'):
        return self.core.region_native(region,t) if output=='native' else self.core.region_voxels(region,t)

    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'):
        return self.core.changes_native(region,ta,tb) if output=='native' else self.core.changes_voxels(region,ta,tb)

    def compact_bytes(self):
        # x0,y0,z0,sx,sy,sz uint16 (12B) + uint32 signature id (4B)
        return self.core.common_storage_bytes()+16*len(self.core.blocks)

    def stats(self): return self.core.stats()
