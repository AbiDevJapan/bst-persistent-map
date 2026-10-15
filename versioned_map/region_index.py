"""Step 10.3: persisted exact-ROI candidate indexes.

``bst_prefix`` parses genuine author-produced 4-axis BST-ID labels and indexes
an independent dyadic prefix on each spatial axis (x,y,f). Query cover uses
exactly the same upstream geospatial quantizer on the ANCHORS of all tiles
that can overlap the integer ROI. Prefix lookup is only a candidate filter:
final inclusion is always computed using authoritative integer tile boxes,
and voxel results are filtered with the original integer predicate.

``morton_hash`` is the conventional strong control: Morton-coded integer tile
coordinates in a hash table, with exact tile coordinate enumeration.

The 'temporal' dimension is a session manifest keyed by UTC date. BST time
payloads are parsed and validated, but there is no special BST temporal
search acceleration here: random access to one named snapshot is O(1).
"""
from __future__ import annotations
import hashlib, json, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .codec import Frame
from .author import author_module

INDEX_MODES = ('scan','bst_prefix','morton_hash')
ZOOMS = (28,28,24,32)
PREFIX_ZOOMS = (22,22,16)


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True,
               separators=(',', ':')).encode('utf-8')).hexdigest()


def _morton3(x,y,z):
    # Exact integer tile coordinates, 21 bits per axis.
    if min(x,y,z) < 0 or max(x,y,z) >= (1 << 21):
        raise ValueError('Morton tile coordinate out of range')
    n=0
    for b in range(21):
        n |= ((x>>b)&1) << (3*b)
        n |= ((y>>b)&1) << (3*b+1)
        n |= ((z>>b)&1) << (3*b+2)
    return n


def _parse_xyz_key(key):
    xyz=tuple(map(int,key.split(',')))
    if len(xyz)!=3 or min(xyz)<0:raise ValueError('invalid manifest tile key')
    return xyz


def _author_payload(label):
    if not label.startswith('official:'):
        raise ValueError('BST prefix mode requires a database built with --codec official')
    _, length, number = label.split(':',2)
    val=int(number,16); bitlen=int(length)
    if val.bit_length()!=bitlen:
        raise ValueError('BST-ID label framing error')
    cell=author_module('region_algebra').Cell.from_id(val)
    if cell.flags!=15 or cell.zooms!=ZOOMS:
        raise ValueError('wrong BST-ID presence/zoom values')
    return cell.values


def _tile_axis_payload(axis, voxel_index, meta):
    frame=Frame(**meta['frame']);size=meta['tile_side']
    x=y=z=0
    indices=[0,0,0];indices[axis]=int(voxel_index)
    x,y,z=indices
    lon,lat,h=frame.coordinates(
        frame.metric_origin_x+(x+.5)*frame.voxel_size,
        frame.metric_origin_y+(y+.5)*frame.voxel_size,
        frame.metric_origin_z+(z+.5)*frame.voxel_size)
    quant=author_module('encoder_tile_calculator')
    if axis==0:
        return quant.encode_x_tile_index(lon, -180., 180., 28)
    if axis==1:
        return quant.encode_y_tile_index(lat, 28)
    return quant.encode_f_tile_index(h,-16383.,16384.,24)


def _tile_range(bounds, dims, side):
    result=[]
    for i in range(3):
        lo=max(0,int(bounds[2*i]));hi=min(int(dims[i]),int(bounds[2*i+1]))
        if hi<=lo:return None
        result.append((lo//side, (hi-1)//side))
    return result

@dataclass
class TileCandidateIndex:
    mode:str
    meta:dict
    manifest_digest:str
    coordinates:np.ndarray
    saved_bytes:int=0
    # For BST: sorted (prefix_value, row number) arrays. For Morton: hash dict.
    prefix_values:tuple=()
    prefix_rows:tuple=()
    morton_lookup:dict|None=None

    @property
    def ram_bytes_estimate(self):
        n=self.coordinates.nbytes
        for a in self.prefix_values+self.prefix_rows:n+=a.nbytes
        # Auxiliary per-row prefix cache and Python hash table are excluded.
        # This is an array lower bound, not process RSS.
        return int(n)

    def keys(self,bounds):
        m=self.meta; side=int(m['tile_side']);dims=m['dims']
        ranges=_tile_range(bounds,dims,side)
        if ranges is None:return [], {'index_probes':0,'prefix_candidates':0,'keys_examined':0}
        if self.mode=='morton_hash':
            result=[];probes=0
            for x in range(ranges[0][0],ranges[0][1]+1):
                for y in range(ranges[1][0],ranges[1][1]+1):
                    for z in range(ranges[2][0],ranges[2][1]+1):
                        probes+=1
                        key=self.morton_lookup.get(_morton3(x,y,z))
                        if key is not None:result.append(key)
            return sorted(result),{'index_probes':probes,'prefix_candidates':len(result),'keys_examined':len(result)}
        if self.mode!='bst_prefix':raise ValueError(self.mode)
        low=[]; high=[]; candidates=[]
        for axis,(tile_lo,tile_hi) in enumerate(ranges):
            # Each overlapping tile has an anchor in this inclusive integer
            # range; the author's quantizers are monotone (Y is reversed).
            v0=_tile_axis_payload(axis,tile_lo*side,m)
            v1=_tile_axis_payload(axis,tile_hi*side,m)
            shift=ZOOMS[axis]-PREFIX_ZOOMS[axis]
            a,b=sorted((v0>>shift,v1>>shift))
            low.append(a);high.append(b)
            values=self.prefix_values[axis]
            L=int(np.searchsorted(values,a,'left'));R=int(np.searchsorted(values,b,'right'))
            candidates.append((R-L,axis,L,R))
        _,axis,L,R=min(candidates)
        indices=self.prefix_rows[axis][L:R]
        if len(indices):
            keep=np.ones(len(indices),bool)
            for j in range(3):
                if j==axis:continue
                # Each prefix array is sorted by its own axis, not row order;
                # reconstruct row->prefix by inexpensive direct access map.
                keep &= (self.prefix_by_row[j][indices]>=low[j])&(self.prefix_by_row[j][indices]<=high[j])
            indices=indices[keep]
            prefix_candidates=len(indices)
            xyz=self.coordinates[indices]
            keep=np.ones(len(indices),bool)
            for j in range(3):
                keep &= (xyz[:,j]>=ranges[j][0])&(xyz[:,j]<=ranges[j][1])
            indices=indices[keep]
        if not len(self.prefix_rows[axis][L:R]):prefix_candidates=0
        keys=[','.join(map(str,xyz)) for xyz in self.coordinates[indices]]
        return sorted(keys),{'index_probes':int(R-L), 'prefix_candidates':prefix_candidates,
                             'keys_examined':int(R-L)}

    @property
    def prefix_by_row(self):
        if not hasattr(self,'_prefix_by_row'):
            vals=[]
            for arr,rows in zip(self.prefix_values,self.prefix_rows):
                buf=np.empty(len(rows),dtype=np.uint32);buf[rows]=arr;vals.append(buf)
            self._prefix_by_row=tuple(vals)
        return self._prefix_by_row


def build_index(store,session,mode='bst_prefix'):
    if mode not in INDEX_MODES[1:]:raise ValueError(mode)
    start=time.perf_counter();manifest=store.all_versions()[session];meta=store.meta()
    if mode=='bst_prefix' and meta['codec']!='official':
        raise ValueError('bst_prefix requires an official BST-ID database')
    keys=sorted(manifest)
    coords=np.asarray([_parse_xyz_key(k) for k in keys],dtype=np.int32).reshape(-1,3)
    arrays={'coordinates':coords,'digest':np.asarray(manifest_hash(manifest)),
            'mode':np.asarray(mode),'session':np.asarray(session),
            'tile_side':np.asarray(meta['tile_side'])}
    if mode=='bst_prefix':
        full=np.asarray([_author_payload(manifest[k]['id']) for k in keys],dtype=np.uint64).reshape(-1,4)
        # Validate temporal labels against session UNIX timestamp to prevent
        # silently mixing maps from different observation sessions.
        import datetime
        t=int(datetime.datetime.fromisoformat(session).replace(tzinfo=datetime.timezone.utc).timestamp())
        if len(full) and not np.all(full[:,3]==t):
            raise ValueError('BST-ID historical time payload mismatch')
        for axis in range(3):
            values=(full[:,axis] >> (ZOOMS[axis]-PREFIX_ZOOMS[axis])).astype(np.uint32)
            order=np.argsort(values,kind='stable').astype(np.int32)
            arrays['v'+str(axis)]=values[order]
            arrays['r'+str(axis)]=order
    else:
        arrays['morton_codes']=np.asarray([_morton3(*c) for c in coords],dtype=np.uint64)
    path=store.root/'indexes'/f'{session}_{mode}.npz';path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**arrays);tmp.replace(path)
    index=load_index(store,session,mode)
    return {'session':session,'mode':mode,'tiles':len(keys),'build_ms':(time.perf_counter()-start)*1000,
            'index_disk_bytes':path.stat().st_size,'array_bytes_estimate':index.ram_bytes_estimate,
            'python_dict_bytes_excluded':mode=='morton_hash'}


def load_index(store,session,mode):
    path=store.root/'indexes'/f'{session}_{mode}.npz'
    if not path.is_file():raise FileNotFoundError(f'build index first: {path}')
    m=store.meta();manifest=store.all_versions()[session]
    with np.load(path,allow_pickle=False) as z:
        if str(z['digest'])!=manifest_hash(manifest) or str(z['session'])!=session or str(z['mode'])!=mode or int(z['tile_side'])!=m['tile_side']:
            raise ValueError('stale or incompatible spatial index: rebuild index')
        coords=z['coordinates'].astype(np.int32,copy=True)
        if len(coords)!=len(manifest):raise ValueError('index row count mismatch')
        vals=tuple(z['v'+str(j)].copy() for j in range(3)) if mode=='bst_prefix' else ()
        rows=tuple(z['r'+str(j)].copy() for j in range(3)) if mode=='bst_prefix' else ()
        lookup=({_morton3(*map(int,c)):','.join(map(str,c)) for c in coords} if mode=='morton_hash' else None)
        if lookup is not None and len(lookup)!=len(coords):raise ValueError('Morton collision')
    return TileCandidateIndex(mode,m,manifest_hash(manifest),coords,path.stat().st_size,vals,rows,lookup)
