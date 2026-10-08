from __future__ import annotations
import numpy as np

from representations.packed_scale import metric_box_to_bounds, _query_linear_rows, temporal_signature_bytes
from .api import OutputMode

class PackedVoxelHistoryMap:
    """Strong packed voxel-history baseline.

    Each ever-observed voxel has one compact spatial code and one shared
    temporal-signature id.  This is intentionally implemented with sorted
    packed arrays rather than a Python dict/hash table so the baseline is not
    penalized by language object overhead.
    """
    name='voxel_history'
    native_kind='voxels'

    def __init__(self,prepared,union_codes,masks):
        self.prepared=prepared; self.spec=prepared.spec; self.n_sessions=len(prepared.session_codes)
        self.codes=union_codes; self.masks=masks
        self.unique_masks=np.unique(masks)
        self.sig_bytes,self.temporal_prefixes=temporal_signature_bytes(self.unique_masks,self.n_sessions)
        self.last_query_stats={}; self.last_change_stats={}

    def _region_candidates(self,region):
        bounds=metric_box_to_bounds(self.spec,region)
        codes=_query_linear_rows(self.codes,self.spec,bounds)
        if not len(codes): return codes,np.empty(0,dtype=np.uint64)
        pos=np.searchsorted(self.codes,codes)
        return codes,self.masks[pos]

    def region_query(self,region,t,*,output:OutputMode='voxels'):
        codes,masks=self._region_candidates(region)
        active=((masks>>np.uint64(t))&1).astype(bool)
        out=codes[active]
        self.last_query_stats={'records_scanned':len(codes),'result_voxels':len(out),'native_primitives':len(out),'native_kind':'voxels'}
        return out

    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'):
        if tb<ta:ta,tb=tb,ta
        codes,masks=self._region_candidates(region)
        A=((masks>>np.uint64(ta))&1).astype(bool); B=((masks>>np.uint64(tb))&1).astype(bool)
        added=codes[(~A)&B]; removed=codes[A&(~B)]
        self.last_change_stats={'records_scanned':len(codes),'result_voxels':len(added)+len(removed),'native_primitives':len(added)+len(removed),'native_kind':'voxels'}
        return added,removed

    def compact_bytes(self):
        # 6-byte logical xyz voxel coordinate + uint32 temporal signature id.
        return 24 + 10*len(self.codes) + self.sig_bytes

    def stats(self):
        return {'voxel_records':len(self.codes),'unique_temporal_signatures':len(self.unique_masks),'temporal_prefixes':self.temporal_prefixes}
