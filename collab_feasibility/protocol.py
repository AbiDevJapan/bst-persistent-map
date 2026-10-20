"""Paired wire-level ROI protocols using the existing Step 10 stores.

Methods:
- voxel_roi: exact uint64 voxel endpoints, region clipped, zlib
- k4_bst: exact K4 occupied boxes, region clipped on the receiver, BST tile selector
- k4_morton: *same* exact K4 occupied boxes and codec, Morton tile selector
- voxel_tiles: full compressed, content-addressed voxel tiles with tile-level reuse

The last method is a deliberately strong baseline: reused tiles need not be resent.
Network byte counts include packet framing, manifest, digests and all transmitted
payload bytes. All methods use the same zlib level and receiver blob cache.
"""
from __future__ import annotations
import hashlib
import json
import struct
import time
import zlib
from dataclasses import dataclass,field
import numpy as np
from realdata.nclt_packed import linear_encode
from versioned_map.store import canonical,_region_filter,_from_bytes

BLOCK_DTYPE=np.dtype([('origin','<u4',(3,)),('exp','u1',(3,))])
METHODS=('voxel_roi','voxel_tiles','k4_morton','k4_bst','k4_tile_morton','k4_tile_bst')


def _digest(blob):return hashlib.sha256(blob).hexdigest()
TILE_ENTRY=struct.Struct('<III32s')  # 12-byte tile xyz + 32-byte SHA256
BLOB_ENTRY=struct.Struct('<32sI')   # SHA256 + compressed payload length


def _pack(meta,payloads):
    """Binary-framed tile and blob manifests (no verbose JSON hashes per tile)."""
    header=dict(meta)
    tiles=header.pop('tiles',[])
    blobs=header.pop('blobs',[])
    header['tile_manifest_count']=len(tiles)
    header['blob_manifest_count']=len(blobs)
    b=json.dumps(header,sort_keys=True,separators=(',',':')).encode('utf8')
    tiles_bin=b''.join(TILE_ENTRY.pack(*map(int,tile['key'].split(',')),bytes.fromhex(tile['hash'])) for tile in tiles)
    blobs_bin=b''.join(BLOB_ENTRY.pack(bytes.fromhex(blob['hash']),int(blob['length'])) for blob in blobs)
    return struct.pack('<I',len(b))+b+tiles_bin+blobs_bin+b''.join(payloads)


def _unpack(message):
    if len(message)<4:raise ValueError('packet too short')
    n=struct.unpack('<I',message[:4])[0]
    if n>len(message)-4:raise ValueError('truncated header')
    meta=json.loads(message[4:4+n]);offset=4+n
    nt=int(meta.pop('tile_manifest_count'))
    nb=int(meta.pop('blob_manifest_count'))
    if nt<0 or nb<0 or (nt*TILE_ENTRY.size+nb*BLOB_ENTRY.size>len(message)-offset):
        raise ValueError('invalid manifests')
    tiles=[]
    for i in range(nt):
        x,y,z,h=TILE_ENTRY.unpack_from(message,offset);offset+=TILE_ENTRY.size
        tiles.append({'key':f'{x},{y},{z}','hash':h.hex()})
    items=[]
    for i in range(nb):
        h,length=BLOB_ENTRY.unpack_from(message,offset);offset+=BLOB_ENTRY.size
        items.append({'hash':h.hex(),'length':length})
    meta['tiles']=tiles;meta['blobs']=items
    payload=memoryview(message)[offset:]
    if sum(int(x['length']) for x in items) != len(payload):
        raise ValueError('payload length mismatch')
    chunks={};pos=0
    for x in items:
        digest=x['hash'];length=int(x['length']);chunk=bytes(payload[pos:pos+length]);pos+=length
        if _digest(chunk)!=digest:raise ValueError('payload SHA256 mismatch')
        chunks[digest]=chunk
    return meta,chunks

def _box_encode(boxes):
    boxes=np.asarray(boxes,dtype=np.int64).reshape(-1,6)
    if len(boxes):
        widths=boxes[:,3:]
        if np.any(widths<1) or np.any(widths & (widths-1)) or np.any(boxes[:,:3]<0):
            raise ValueError('K4 box not positive dyadic anchored coordinates')
    out=np.empty(len(boxes),dtype=BLOCK_DTYPE)
    out['origin']=boxes[:,:3].astype('<u4')
    out['exp']=np.log2(boxes[:,3:]).astype(np.uint8)
    return zlib.compress(out.tobytes(),6)


def _box_decode(blob):
    raw=zlib.decompress(blob)
    if len(raw)%BLOCK_DTYPE.itemsize:raise ValueError('invalid packed box array')
    items=np.frombuffer(raw,dtype=BLOCK_DTYPE)
    if not len(items):return np.empty((0,6),dtype=np.int64)
    return np.column_stack((items['origin'].astype(np.int64), 1<<items['exp'].astype(np.int64)))


def _expand_clipped(boxes,bounds,spec,max_voxels=3_000_000):
    """Exact ROI intersection with vectorized occupied-cell expansion."""
    boxes=np.asarray(boxes,dtype=np.int64).reshape(-1,6)
    if not len(boxes):return np.empty(0,dtype=np.uint64)
    lower=np.maximum(np.asarray(bounds[::2],dtype=np.int64),0)
    upper=np.minimum(np.asarray(bounds[1::2],dtype=np.int64),
                     np.asarray([spec.nx,spec.ny,spec.nz],dtype=np.int64))
    lo=np.maximum(boxes[:,:3],lower[None,:])
    hi=np.minimum(boxes[:,:3]+boxes[:,3:],upper[None,:])
    lengths=np.maximum(hi-lo,0)
    n=np.prod(lengths,axis=1)
    total=int(n.sum())
    if total>max_voxels:raise MemoryError('ROI decoded voxel limit reached')
    if not total:return np.empty(0,dtype=np.uint64)
    active=n>0
    lo=lo[active];lengths=lengths[active];n=n[active]
    row=np.repeat(np.arange(len(n)),n)
    starts=np.cumsum(n)-n
    offset=np.arange(total,dtype=np.int64)-np.repeat(starts,n)
    dy=lengths[row,1];dz=lengths[row,2]
    x=lo[row,0]+offset//(dy*dz)
    y=lo[row,1]+(offset//dz)%dy
    z=lo[row,2]+offset%dz
    result=canonical(linear_encode(x,y,z,spec))
    if len(result)!=total:raise ValueError('overlapping blocks or repeated occupancy')
    return result

@dataclass
class Receiver:
    blobs:dict=field(default_factory=dict)

    def apply(self,message,spec,expected_bounds=None):
        start=time.perf_counter()
        meta,received=_unpack(message)
        bounds=tuple(meta['bounds'])
        if expected_bounds is not None and bounds!=tuple(expected_bounds):raise ValueError('ROI header mismatch')
        for digest,blob in received.items():self.blobs[digest]=blob
        name=meta['method']
        if name in ('voxel_tiles','k4_tiles'):
            parts=[]
            for tile in meta['tiles']:
                digest=tile['hash']
                if digest not in self.blobs:raise ValueError('receiver lacks required tile')
                raw=self.blobs[digest]
                if _digest(raw)!=digest:raise ValueError('cached tile corrupt')
                if name=='voxel_tiles':parts.append(_region_filter(_from_bytes(zlib.decompress(raw)),spec,bounds))
                else:parts.append(_expand_clipped(_box_decode(raw),bounds,spec))
            result=canonical(np.concatenate(parts) if parts else np.empty(0,dtype=np.uint64))
        else:
            digest=meta['roi_hash']
            if digest not in self.blobs:raise ValueError('receiver lacks required ROI')
            blob=self.blobs[digest]
            if _digest(blob)!=digest:raise ValueError('cached ROI corrupt')
            if name=='voxels':result=_from_bytes(zlib.decompress(blob))
            elif name=='k4_boxes':
                result=_expand_clipped(_box_decode(blob),bounds,spec)
            else:raise ValueError('unknown protocol')
        return result,(time.perf_counter()-start)*1000


class Sender:
    def __init__(self,voxel_store,geometry_store):
        self.store=voxel_store;self.geometry=geometry_store

    def send(self,method,version,bounds,receiver):
        if method not in METHODS:raise ValueError(method)
        start=time.perf_counter()
        stats={};payloads=[];items=[]
        wire_method=({'voxel_roi':'voxels','k4_bst':'k4_boxes','k4_morton':'k4_boxes',
                      'k4_tile_bst':'k4_tiles','k4_tile_morton':'k4_tiles'}.get(method,method))
        meta={'schema':1,'method':wire_method,'version':version,'bounds':list(map(int,bounds)),'blobs':items}
        if method=='voxel_tiles':
            manifest=self.store.all_versions()[version]
            keys=self.store._matching_keys(manifest,bounds,session=version,index_mode='morton_hash')
            meta['tiles']=[{'key':k,'hash':manifest[k]['hash']} for k in sorted(keys)]
            for k in sorted(keys):
                digest=manifest[k]['hash']
                if digest not in receiver.blobs and digest not in (b['hash'] for b in items):
                    packed=(self.store.root/'blobs'/digest).read_bytes()
                    if _digest(packed)!=digest:raise IOError('source tile SHA256 mismatch')
                    payloads.append(packed);items.append({'hash':digest,'length':len(packed)})
            stats['primitives']=sum(manifest[k]['count'] for k in keys)
            stats['tiles']=len(keys)
        elif method.startswith('k4_tile_'):
            mode='bst' if method.endswith('bst') else 'morton'
            keys,probes=self.geometry._candidate_keys(version,bounds,mode)
            keys=sorted(keys)
            meta['tiles']=[]
            active_bit=np.uint32(1)<<np.uint32(self.geometry.meta['sessions'].index(version))
            for key in keys:
                cells=self.geometry._read_tile(key)
                active=cells[(cells['versions']&active_bit)!=0]
                if not len(active):continue
                from versioned_map.region_index import _parse_xyz_key
                base=np.asarray(_parse_xyz_key(key),dtype=np.int64)*int(self.geometry.meta['tile_side'])
                boxes=np.column_stack((base[None,:]+active['xyz'].astype(np.int64),1<<active['exp'].astype(np.int64)))
                boxes=boxes[np.lexsort(tuple(boxes[:,j] for j in reversed(range(6))))]
                compressed=_box_encode(boxes)
                digest=_digest(compressed)
                meta['tiles'].append({'key':key,'hash':digest})
                if digest not in receiver.blobs and digest not in (b['hash'] for b in items):
                    payloads.append(compressed);items.append({'hash':digest,'length':len(compressed)})
                stats['primitives']=stats.get('primitives',0)+len(boxes)
            stats['tiles']=len(meta['tiles'])
            stats['index_probes']=probes
        else:
            if method=='voxel_roi':
                vox,query_stats=self.store.query(version,bounds,stats=True,index_mode='morton_hash')
                packed=zlib.compress(np.asarray(vox,dtype='<u8').tobytes(),6)
                stats['primitives']=len(vox)
            else:
                mode='bst' if method=='k4_bst' else 'morton'
                blocks,query_stats=self.geometry.primitives(version,bounds,mode=mode,stats=True)
                # Force stable ordering, ensuring the same wire geometry for both indexes.
                if len(blocks):blocks=blocks[np.lexsort(tuple(blocks[:,j] for j in reversed(range(6))))]
                packed=_box_encode(blocks)
                stats['primitives']=len(blocks)
            digest=_digest(packed)
            meta['roi_hash']=digest
            if digest not in receiver.blobs:
                items.append({'hash':digest,'length':len(packed)})
                payloads.append(packed)
            stats.update({'index_ms':query_stats.get('index_ms',0),
                          'index_probes':query_stats.get('index_probes',query_stats.get('keys_examined',0))})
        encoded=_pack(meta,payloads)
        stats.update({'request_select_encode_ms':(time.perf_counter()-start)*1000,
                      'packet_bytes':len(encoded),'body_bytes':sum(map(len,payloads)),
                      'header_bytes':len(encoded)-sum(map(len,payloads)),
                      'blob_cache_hits':(len(meta.get('tiles',[]))-len(items) if method=='voxel_tiles' else int(len(items)==0))})
        return encoded,stats
