from __future__ import annotations

from representations.step83.api import OutputMode
from .codec import PrototypeBSTSpatialCodec


class _BSTBlockMapBase:
    native_kind='blocks'
    native_change_exact=True

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


class BSTCompactMap(_BSTBlockMapBase):
    """Step-8.3 full-history-conditioned adaptive map."""
    name='bst_compact_bucket'


class BSTTimeLocalMap(_BSTBlockMapBase):
    """Step-8.4 time-local adaptive map with isolated BST spatial codec.

    The construction/query core is BST-agnostic.  Swapping in a verified BST-ID
    codec still requires changing only the codec implementation supplied here.
    """
    name='bst_time_local'
    native_change_exact=False


class BSTTimeLocalTimeIndexedMap(_BSTBlockMapBase):
    """Step-8.5 time-local map using counted (spatial bucket, session) postings.

    The block table and BST codec are identical to :class:`BSTTimeLocalMap`;
    only the auxiliary runtime index differs.  Its storage is counted.
    """
    name='bst_time_local_tindex'
    native_change_exact=False

    def region_query(self,region,t,*,output:OutputMode='voxels'):
        if output=='native':
            return self.core.region_native(region,t,index_mode='time')
        return self.core.region_voxels(region,t,index_mode='time')

    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'):
        if output=='native':
            return self.core.changes_native(region,ta,tb)
        return self.core.changes_voxels(region,ta,tb,index_mode='time')

    def compact_bytes(self):
        return self.core.common_storage_bytes(index_mode='time') + self.codec.spatial_table_bytes(self.core.blocks,self.core.spec) + 4*len(self.core.blocks)

    def stats(self):
        out=dict(self.core.stats(index_mode='time'))
        out['bst_codec']=self.codec.name
        return out
