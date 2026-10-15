from __future__ import annotations
import numpy as np

from representations.packed_scale import metric_box_to_bounds, _query_linear_rows
from .api import OutputMode

class DeltaVersionMap:
    """Latest-base + reverse positive/negative deltas.

    The latest session is the materialized base.  For k=1..N-1, reverse delta
    k transforms session k -> session k-1:
       add_to_previous    = M[k-1] - M[k]
       remove_to_previous = M[k]   - M[k-1]

    Historical regional queries apply only the region-filtered reverse deltas.
    This is deliberately simple and strong enough to represent the common
    base+delta/version-control strategy without claiming to reproduce any one
    published system exactly.
    """
    name='delta_version'
    native_kind='voxels'

    def __init__(self,prepared):
        self.prepared=prepared; self.spec=prepared.spec; self.n_sessions=len(prepared.session_codes)
        self.base=prepared.session_codes[-1]
        self.reverse_add=[None]*self.n_sessions
        self.reverse_remove=[None]*self.n_sessions
        total=0
        for k in range(1,self.n_sessions):
            prev=prepared.session_codes[k-1]; cur=prepared.session_codes[k]
            add=np.setdiff1d(prev,cur,assume_unique=True)
            rem=np.setdiff1d(cur,prev,assume_unique=True)
            self.reverse_add[k]=add; self.reverse_remove[k]=rem
            total += len(add)+len(rem)
        self.delta_records=total
        self.last_query_stats={}; self.last_change_stats={}

    def _region(self,arr,region):
        return _query_linear_rows(arr,self.spec,metric_box_to_bounds(self.spec,region))

    def region_query(self,region,t,*,output:OutputMode='voxels'):
        state=self._region(self.base,region)
        delta_scanned=0
        for k in range(self.n_sessions-1,t,-1):
            add=self._region(self.reverse_add[k],region); rem=self._region(self.reverse_remove[k],region)
            delta_scanned += len(add)+len(rem)
            if len(add): state=np.union1d(state,add)
            if len(rem): state=np.setdiff1d(state,rem,assume_unique=True)
        self.last_query_stats={'delta_records_scanned':delta_scanned,'records_scanned':len(state)+delta_scanned,'result_voxels':len(state),'native_primitives':len(state),'native_kind':'voxels','reverse_steps':self.n_sessions-1-t}
        return state

    def changes_query(self,region,ta,tb,*,output:OutputMode='voxels'):
        if tb<ta:ta,tb=tb,ta
        a=self.region_query(region,ta); sa=dict(self.last_query_stats)
        b=self.region_query(region,tb); sb=dict(self.last_query_stats)
        added=np.setdiff1d(b,a,assume_unique=True); removed=np.setdiff1d(a,b,assume_unique=True)
        self.last_change_stats={'records_scanned':sa.get('records_scanned',0)+sb.get('records_scanned',0),'delta_records_scanned':sa.get('delta_records_scanned',0)+sb.get('delta_records_scanned',0),'result_voxels':len(added)+len(removed),'native_primitives':len(added)+len(removed),'native_kind':'voxels'}
        return added,removed

    def compact_bytes(self):
        # Same logical voxel coordinate accounting as snapshots: 6 bytes/voxel.
        # 8B base header + 16B per transition for add/remove counts/offsets.
        return 8 + 6*len(self.base) + 16*(self.n_sessions-1) + 6*self.delta_records

    def stats(self):
        return {'base_voxels':len(self.base),'delta_records':self.delta_records,'transitions':self.n_sessions-1,'anchor':'latest'}
