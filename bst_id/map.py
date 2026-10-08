from __future__ import annotations

from representations.step83.api import OutputMode
from .codec import PrototypeBSTSpatialCodec

class BSTCompactMap:
    """Thin BST-specific wrapper over a generic AdaptiveBlockCore."""
    name='bst_compact_bucket'
    native_kind='blocks'

    def __init__(self,core,codec=None):
        self.core=core
        self.codec=codec or PrototypeBSTSpatialCodec()

    @property
    def last_query_stats(self): return self.core.last_query_stats
    @property
    def last_change_stats(self): return self.core.last_change_stats

    def region_query(self,region,t,*,output:OutputMode='voxels'):
        return self.core.region_native(region,t) if output=='native' else self.core.region_voxels(region,t)

    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'):
        return self.core.changes_native(region,ta,tb) if output=='native' else self.core.changes_voxels(region,ta,tb)

    def compact_bytes(self):
        # Signature ID remains generic uint32 map metadata. Only spatial block
        # addressing is delegated to the replaceable BST codec.
        return self.core.common_storage_bytes() + self.codec.spatial_table_bytes(self.core.blocks,self.core.spec) + 4*len(self.core.blocks)

    def stats(self):
        out=dict(self.core.stats())
        out['bst_codec']=self.codec.name
        return out
