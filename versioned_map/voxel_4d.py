"""Optional Step 10.5A *voxel occupancy* compression via exact 4D sibling merging.

Unlike aggregate_4d.py (which merges the presence of NONEMPTY TILES while
keeping per-tile geometry), this experimental representation directly stores
occupied VOXEL cells in BST-ID's local integer x/y/f and UNIX-time domain.
It preserves the exact sets of observed endpoints at the five version times.
It is deliberately region-bounded to avoid unbounded Python object materialization
when ingesting millions of sparse NCLT endpoints.

All merging requires both dyadic siblings' occupancy, for the entire covered
space/time cell. A mere lack of evidence does NOT mean observed free space.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
import time
import numpy as np

from .aggregate_4d import SPATIAL_BITS, _merge_exact_siblings, _ranges
from .author import author_module
from .interval_store import _dyadic_time_cover, _timestamp, MAX_TIME
from .store import VersionedStore, canonical
from realdata.nclt_packed import linear_decode,linear_encode
from synthetic.world import WorldSpec


def _spec(meta):
    return WorldSpec(*[int(v)*float(meta['voxel_size_m']) for v in meta['dims']],
                     float(meta['voxel_size_m']))


class Voxel4D:
    def __init__(self, path):
        self.path=Path(path)
        self._data=None

    @classmethod
    def build(cls,source,path,bounds,*,max_atoms=500_000,merged=True):
        source=source if isinstance(source,VersionedStore) else VersionedStore(source)
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        names=list(source.all_versions())
        meta=source.meta();spec=_spec(meta)
        bounds=tuple(int(v) for v in bounds)
        if len(bounds)!=6 or any(bounds[i]>=bounds[i+1] for i in (0,2,4)):
            raise ValueError('invalid spatial ROI')
        time0=time.perf_counter()
        spans={}
        for i,s in enumerate(names):
            codes=source.query(s,bounds,index_mode='scan')
            if len(codes)>max_atoms:
                raise ValueError('region exceeds max_atoms per session; use smaller ROI')
            for code in codes:
                spans.setdefault(int(code),[]).append(i)
        raw=set();nruns=0
        for code,versions in spans.items():
            x,y,z=(int(a[0]) for a in linear_decode(np.asarray([code],dtype=np.uint64),spec))
            if min(x,y,z)<0 or max(x,y,z)>=(1<<SPATIAL_BITS):
                raise ValueError('voxel address outside 21-bit local domain')
            runs=[];start=previous=versions[0]
            for version in versions[1:]:
                if version != previous+1:
                    runs.append((start,previous+1))
                    start=version
                previous=version
            runs.append((start,previous+1))
            nruns+=len(runs)
            for a,b in runs:
                lo=_timestamp(names[a]);hi=_timestamp(names[b]) if b<len(names) else MAX_TIME
                for zoom,prefix in _dyadic_time_cover(lo,hi):
                    raw.add((SPATIAL_BITS,x,SPATIAL_BITS,y,SPATIAL_BITS,z,zoom,prefix))
                    if len(raw)>max_atoms:
                        raise ValueError('region has too many unmerged space-time atoms; '
                                         'lower ROI or increase --max-atoms carefully')
        nodes,merges = _merge_exact_siblings(raw) if merged else (raw,{'merged_pairs_by_axis':
            dict.fromkeys(('x','y','z','t'),0),'merge_rounds':0})
        arr=np.asarray(sorted(nodes),dtype=np.uint32).reshape(-1,8)
        Cell=author_module('region_algebra').Cell
        ids=np.empty(len(arr),dtype='V17')
        for k,c in enumerate(arr):
            obj=Cell.from_parts(x=(int(c[0]),int(c[1])),y=(int(c[2]),int(c[3])),
                                f=(int(c[4]),int(c[5])),t=(int(c[6]),int(c[7])))
            ids[k]=np.void(obj.to_id().to_bytes(17,'big'))
        checksum=sha256((source.root/'versions.json').read_bytes()).hexdigest()
        metadata={'schema':1,'sessions':names,'bounds':bounds,'source_dims':meta['dims'],
                  'voxel_size_m':meta['voxel_size_m'],'frame':'local voxel coords + UTC Unix seconds',
                  'semantics':'exact occupied endpoint observations at each stored version',
                  'source_versions_sha256':checksum,'merged':bool(merged),
                  'input_unique_voxels':len(spans),'presence_runs':nruns,
                  'raw_4d_cells':len(raw),'final_4d_cells':len(nodes),
                  'merge_stats':merges,'build_ms':(time.perf_counter()-time0)*1000}
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,cells=arr,bst_ids=ids,
                            metadata=np.asarray(json.dumps(metadata)))
        return cls(path),metadata

    def load(self):
        if self._data is None:
            with np.load(self.path,allow_pickle=False) as z:
                cells=z['cells'].copy();ids=z['bst_ids'].copy();meta=json.loads(str(z['metadata']))
            if len(cells)!=len(ids) or cells.ndim!=2 or cells.shape[1]!=8:
                raise IOError('invalid voxel 4D index')
            Cell=author_module('region_algebra').Cell
            for i,c in enumerate(cells):
                obj=Cell.from_parts(x=(int(c[0]),int(c[1])),y=(int(c[2]),int(c[3])),
                                    f=(int(c[4]),int(c[5])),t=(int(c[6]),int(c[7])))
                if obj.to_id().to_bytes(17,'big') != bytes(ids[i]):
                    raise IOError('corrupt voxel 4D ID')
            self._data=(cells,_ranges(cells),meta)
        return self._data

    def query(self,session,bounds,*,stats=False,max_output_voxels=5_000_000):
        start=time.perf_counter()
        cells,box,meta=self.load()
        if session not in meta['sessions']:raise KeyError(session)
        if len(bounds)!=6 or any(bounds[j]>=bounds[j+1] for j in (0,2,4)):
            raise ValueError('invalid ROI bounds')
        bbox=[(max(int(bounds[i]),int(meta['bounds'][i]),0),
               min(int(bounds[i+1]),int(meta['bounds'][i+1]),int(meta['source_dims'][i//2])))
              for i in (0,2,4)]
        if any(a>=b for a,b in bbox):
            empty=np.empty(0,dtype=np.uint64)
            return (empty,{'cells_examined':len(cells),'cells_hit':0,'latency_ms':(time.perf_counter()-start)*1000}) if stats else empty
        t=_timestamp(session)
        mask=(box[:,6]<=t)&(box[:,7]>t)
        for axis,(lo,hi) in enumerate(bbox):
            mask &= (box[:,2*axis]<hi)&(box[:,2*axis+1]>lo)
        ix=np.flatnonzero(mask)
        xyz=[];voxel_count=0
        for k in ix:
            b=box[k]
            spans=[np.arange(max(int(b[2*a]),bbox[a][0]),min(int(b[2*a+1]),bbox[a][1]),dtype=np.uint64)
                   for a in range(3)]
            length=int(len(spans[0])*len(spans[1])*len(spans[2]))
            voxel_count+=length
            if voxel_count>max_output_voxels:
                raise MemoryError('expanded ROI exceeds max_output_voxels')
            if length:
                gx,gy,gz=np.meshgrid(*spans,indexing='ij')
                xyz.append(linear_encode(gx.ravel(),gy.ravel(),gz.ravel(),
                       WorldSpec(*[int(v)*float(meta['voxel_size_m']) for v in meta['source_dims']],
                                 float(meta['voxel_size_m']))))
        result=canonical(np.concatenate(xyz) if xyz else np.empty(0,dtype=np.uint64))
        info={'cells_examined':len(cells),'cells_hit':len(ix), 'expanded_voxels':voxel_count,
              'latency_ms':(time.perf_counter()-start)*1000, 'replayed_transitions':0}
        return (result,info) if stats else result
