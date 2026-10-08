from __future__ import annotations
from .api import OutputMode

class SnapshotAdapter:
    name='snapshot'; native_kind='voxels'
    def __init__(self,impl): self.impl=impl
    @property
    def last_query_stats(self): return self.impl.last_query_stats
    @property
    def last_change_stats(self): return self.impl.last_change_stats
    def region_query(self,region,t,*,output:OutputMode='voxels'): return self.impl.region_codes(region,t)
    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'): return self.impl.changes_codes(region,ta,tb)
    def compact_bytes(self): return self.impl.compact_bytes()
    def stats(self): return {'sessions':len(self.impl.prepared.session_codes)}

class OctreeAdapter:
    name='octree_shared_sparse'; native_kind='voxel_leaves'
    def __init__(self,impl): self.impl=impl
    @property
    def last_query_stats(self): return self.impl.last_query_stats
    @property
    def last_change_stats(self): return self.impl.last_change_stats
    def region_query(self,region,t,*,output:OutputMode='voxels'): return self.impl.region_codes(region,t)
    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'): return self.impl.changes_codes(region,ta,tb)
    def compact_bytes(self): return self.impl.compact_bytes()
    def stats(self): return self.impl.stats()
