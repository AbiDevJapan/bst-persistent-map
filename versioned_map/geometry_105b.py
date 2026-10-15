"""Step 10.5B: build-time *geometry* aggregation with exact 4D version membership.

Unlike the Step 10.5A tile-presence index, each block is a true occupied
volume: all finest-resolution voxel positions covered by the block were
observed in the indicated map version. Spatial siblings are coalesced to a
configurable maximum K (= number of 0.5m finest voxels per block). Identical
spatial block descriptors across historical versions share a single mask.

The original point clouds / per-voxel surfel moments are NOT stored in Step 10.
This is lossless relative to the *voxelized endpoint sets* and must not be
misrepresented as the old Step-9 planarity/normal-gated surfel compression.

All modes share *identical* compressed geometry, and compare alternative
4D BST-ID tile selection with a Morton tile hash. Original author code is
unmodified. Cells use a documented LOCAL unsigned 21-bit voxel coordinate
frame; the official Cell API supplies the x/y/f/t prefix algebra.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil
import time
import zlib
import numpy as np

from realdata.nclt_packed import linear_decode, linear_encode
from .store import VersionedStore, _from_bytes, canonical
from .interval_store import _timestamp, _dyadic_time_cover, MAX_TIME
from .region_index import _morton3, _tile_range, _parse_xyz_key
from .author import author_module

# Each stored block is x,y,z local uint8; dyadic lengths as exponent uint8;
# version mask uint32. 10 bytes before zlib. No duplication by session.
BLOCK_DTYPE=np.dtype([('xyz','u1',(3,)),('exp','u1',(3,)),('versions','<u4')])
INDEX_VERSION=1


def _block_key(a, side):
    bits=max(1,(side-1).bit_length())
    shifts=(0,bits,2*bits,3*bits,3*bits+4,3*bits+8)
    # origin xyz and exponents are disjoint fields
    result=np.zeros(len(a),dtype=np.uint64)
    for j in range(3):result |= a[:,j].astype(np.uint64)<<np.uint64(shifts[j])
    for j in range(3):result |= (np.log2(a[:,j+3]).astype(np.uint64))<<np.uint64(shifts[j+3])
    return result


def merge_occupied_voxels(local_xyz,side=32,k=4):
    """Exact dyadic sibling merges; return non-overlapping occupied boxes.

    A merged box requires BOTH children. K caps fine-voxel support; with K=4
    each block spans at most four original observed voxels. This implementation
    never fills gaps, averages coordinates, or suppresses observations.
    """
    if side>256 or side<=0 or side & (side-1):raise ValueError('tile side must be a power of 2 <=256')
    if k<1 or k>side**3:raise ValueError('K must be between 1 and tile capacity')
    xyz=np.asarray(local_xyz,dtype=np.int32).reshape(-1,3)
    if len(xyz) and (np.any(xyz<0) or np.any(xyz>=side)):
        raise ValueError('voxel outside tile')
    if len(xyz) and len(np.unique(xyz,axis=0))!=len(xyz):
        raise ValueError('duplicate voxel center')
    blocks=np.empty((len(xyz),6),dtype=np.int32)
    blocks[:,:3]=xyz;blocks[:,3:]=1
    if k==1:return blocks
    bits=max(1,(side-1).bit_length())
    shifts=(0,bits,2*bits)
    while True:
        nmerged=0
        for ax in range(3):
            if len(blocks)<2:break
            keys=_block_key(blocks,side)
            order=np.argsort(keys,kind='stable');sk=keys[order]
            o=blocks[:,ax];width=blocks[:,ax+3]
            cap=np.prod(blocks[:,3:].astype(np.int64),axis=1)*2<=k
            eligible=(o%(2*width)==0)&(o+2*width<=side)&cap
            lows=np.flatnonzero(eligible)
            if not len(lows):continue
            wanted=keys[lows]+(width[lows].astype(np.uint64)<<np.uint64(shifts[ax]))
            pos=np.searchsorted(sk,wanted)
            ok=pos<len(sk);ok[ok]=sk[pos[ok]]==wanted[ok]
            lows=lows[ok]; sibs=order[pos[ok]]
            if not len(lows):continue
            consumed=np.zeros(len(blocks),dtype=bool)
            consumed[lows]=True;consumed[sibs]=True
            new=blocks[lows].copy();new[:,ax+3]*=2
            blocks=np.concatenate((blocks[~consumed],new),axis=0)
            nmerged+=len(lows)
        if not nmerged:break
    return blocks


def coalesce_time_coherent_siblings(local_masks,side,k):
    """Second build-time XYZ pass over the *temporal* geometry representation.

    A pair may merge only if BOTH occupied spatial siblings have the SAME
    historical version mask, and the parent would contain at most K voxels.
    This makes spatial aggregation conditional on the complete time history,
    rather than only on one mapping session at a time.
    """
    if len(local_masks)<2:return dict(local_masks),0
    a=np.asarray(list(local_masks),dtype=np.int32).reshape(-1,6)
    mask=np.asarray(list(local_masks.values()),dtype=np.uint32)
    bit=max(1,(side-1).bit_length());shifts=(0,bit,2*bit)
    total=0
    while True:
        count=0
        for axis in range(3):
            if len(a)<2:break
            keys=_block_key(a,side)
            order=np.argsort(keys,kind='stable');sorted_keys=keys[order]
            origin=a[:,axis];width=a[:,axis+3]
            support=np.prod(a[:,3:],axis=1,dtype=np.int64)*2<=k
            lower=(origin%(2*width)==0)&(origin+2*width<=side)&support
            ix=np.flatnonzero(lower)
            if not len(ix):continue
            sib=keys[ix]+(width[ix].astype(np.uint64)<<np.uint64(shifts[axis]))
            pos=np.searchsorted(sorted_keys,sib)
            valid=pos<len(sorted_keys)
            valid[valid]=sorted_keys[pos[valid]]==sib[valid]
            ix=ix[valid];j=order[pos[valid]]
            valid=mask[ix]==mask[j]
            ix=ix[valid];j=j[valid]
            if not len(ix):continue
            consumed=np.zeros(len(a),dtype=bool);consumed[ix]=True;consumed[j]=True
            parent=a[ix].copy();parent[:,axis+3]*=2
            a=np.concatenate((a[~consumed],parent))
            mask=np.concatenate((mask[~consumed],mask[ix]))
            count+=len(ix)
        total+=count
        if count==0:break
    return {tuple(map(int,row)):int(v) for row,v in zip(a,mask)},total


def _cell_struct(blocks, local_to_mask):
    out=np.empty(len(local_to_mask),dtype=BLOCK_DTYPE)
    for j,(block,mask) in enumerate(sorted(local_to_mask.items())):
        a=np.asarray(block,dtype=np.int32)
        out['xyz'][j]=a[:3].astype(np.uint8)
        out['exp'][j]=np.log2(a[3:]).astype(np.uint8)
        out['versions'][j]=np.uint32(mask)
    return out


def _runs(mask, n):
    a=None
    for i in range(n+1):
        on=i<n and bool(int(mask)&(1<<i))
        if on and a is None:a=i
        elif not on and a is not None:
            yield a,i
            a=None


def _put_json(p,value):
    p=Path(p); tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,separators=(',',':')));tmp.replace(p)


class Geometry4D:
    """Shared 4D geometry blocks + alternative BST or Morton spatial index."""
    def __init__(self,root):
        self.root=Path(root)
        self._meta=None;self._directory=None;self._index={}

    @property
    def meta(self):
        if self._meta is None:self._meta=json.loads((self.root/'metadata.json').read_text())
        return self._meta

    @property
    def directory(self):
        if self._directory is None:self._directory=json.loads((self.root/'tiles.json').read_text())
        return self._directory

    @classmethod
    def build(cls, source,root,*,k=4,through_session=None):
        src=source if isinstance(source,VersionedStore) else VersionedStore(source)
        versions=src.all_versions();sessions=list(versions)
        if through_session is not None:
            if through_session not in sessions:raise KeyError(through_session)
            sessions=sessions[:sessions.index(through_session)+1]
        if not sessions or len(sessions)>32:raise ValueError('requires 1..32 historical sessions')
        m=src.meta();side=int(m['tile_side']);dims=[int(d) for d in m['dims']]
        if side>256:raise ValueError('tile side >256 is not implemented')
        if any(int(d) >= 1<<21 for d in dims):raise ValueError('21-bit voxel frame exceeded')
        if any(_timestamp(a)>=_timestamp(b) for a,b in zip(sessions,sessions[1:])):
            raise ValueError('sessions must be strictly chronological')
        dest=cls(root)
        if dest.root.exists():raise FileExistsError(root)
        dest.root.mkdir(parents=True);(dest.root/'blobs').mkdir()
        t0=time.perf_counter()
        spec=src._spec()
        all_keys=sorted(set().union(*(set(versions[s]) for s in sessions)))
        sums={'input_voxels_across_sessions':0,'spatial_blocks_across_sessions':0,
              'distinct_spatial_blocks_over_time':0,'block_version_runs':0,
              'tiles_with_geometry':0,'spatial_merge_count':0,'joint_xyz_time_merge_count':0}
        entries={}
        try:
            for key in all_keys:
                tile=np.asarray(_parse_xyz_key(key),dtype=np.int64)
                anchor=tile*side
                masks={}; source_to_blocks={}
                for ti,s in enumerate(sessions):
                    rec=versions[s].get(key)
                    if rec is None:continue
                    digest=rec['hash']
                    # Cache within tile when repeated source contents are identical.
                    if digest not in source_to_blocks:
                        packed=(src.root/'blobs'/digest).read_bytes()
                        if sha256(packed).hexdigest()!=digest:
                            raise IOError('source compressed voxel blob hash mismatch')
                        vox=_from_bytes(zlib.decompress(packed))
                        x,y,z=linear_decode(vox,spec)
                        xyz=np.column_stack((x,y,z)).astype(np.int64)-anchor
                        result=merge_occupied_voxels(xyz,side,k)
                        if sum(int(np.prod(row[3:])) for row in result)!=len(vox):
                            raise AssertionError('spatial cell merges changed the number of occupied voxels')
                        source_to_blocks[digest]=(len(vox),[tuple(int(n) for n in row) for row in result])
                    original_count,blocks=source_to_blocks[digest]
                    sums['input_voxels_across_sessions']+=original_count
                    sums['spatial_blocks_across_sessions']+=len(blocks)
                    sums['spatial_merge_count']+=original_count-len(blocks)
                    for block in blocks:masks[block]=masks.get(block,0)|(1<<ti)
                if not masks:continue
                masks,extra=coalesce_time_coherent_siblings(masks,side,k)
                sums['joint_xyz_time_merge_count']+=extra
                cells=_cell_struct(None,masks)
                compressed=zlib.compress(cells.tobytes(),level=6)
                digest=sha256(compressed).hexdigest()
                blob=dest.root/'blobs'/digest
                if not blob.exists():blob.write_bytes(compressed)
                entries[key]={'hash':digest,'cells':len(cells)}
                sums['tiles_with_geometry']+=1
                sums['distinct_spatial_blocks_over_time']+=len(cells)
                sums['block_version_runs']+=sum(1 for v in cells['versions'] for _ in _runs(v,len(sessions)))
            _put_json(dest.root/'tiles.json',entries)
            _put_json(dest.root/'metadata.json',{
                'schema':INDEX_VERSION,'source_codec':m['codec'], 'source_meta':m,
                'sessions':sessions,'k':int(k),'tile_side':side,
                'semantics':'EXACT observed-endpoint occupancy; each block covers only fully observed voxels at each stored version',
                'not_surfel_gate':'no original scan covariance/normals in Step 10 voxel store; no planarity approximation',
                'frame':'local nonnegative canonical voxels, 21-bit x/y/z and UTC UNIX seconds t',
                'stats':sums,'build_seconds':time.perf_counter()-t0})
            dest.rebuild_indexes()
            return dest
        except BaseException:
            shutil.rmtree(dest.root,ignore_errors=True)
            raise

    def append_from_store(self,source,target_session,*,gc=False):
        """Append one version without re-aggregating earlier source sessions.

        Read only incoming snapshot tile payloads. Existing 4D spatial blocks
        are extended through one additional time bit when they recur. Blob
        writes are content-addressed; an index rebuild is currently required.
        """
        src=source if isinstance(source,VersionedStore) else VersionedStore(source)
        sessions=self.meta['sessions'];source_sessions=list(src.all_versions())
        if (len(sessions)>=len(source_sessions) or
            source_sessions[:len(sessions)]!=sessions or
            target_session!=source_sessions[len(sessions)]):
            raise ValueError('append requires the next chronological source session')
        if len(sessions)>=32:raise ValueError('32-session bitmask limit reached')
        m=src.meta();old=self.meta['source_meta']
        if any(m[x]!=old[x] for x in ('dims','tile_side','voxel_size_m')):
            raise ValueError('incompatible source geometry')
        ti=len(sessions);side=int(self.meta['tile_side']);k=int(self.meta['k'])
        spec=src._spec();before=dict(self.directory);updates={}
        stats=dict(self.meta['stats']);added_blocks=0;changed_tiles=0
        for key,entry in src.all_versions()[target_session].items():
            base=np.asarray(_parse_xyz_key(key),dtype=np.int64)*side
            blob=(src.root/'blobs'/entry['hash']).read_bytes()
            if sha256(blob).hexdigest()!=entry['hash']:raise IOError('source voxel blob corrupted')
            codes=_from_bytes(zlib.decompress(blob));x,y,z=linear_decode(codes,spec)
            xyz=np.column_stack((x,y,z)).astype(np.int64)-base
            blocks=merge_occupied_voxels(xyz,side,k)
            if sum(int(np.prod(c[3:])) for c in blocks)!=len(codes):
                raise AssertionError('append geometry did not preserve voxel occupancy')
            masks={}
            if key in before:
                for c in self._read_tile(key):
                    width=1<<c['exp'].astype(np.int64)
                    cell=tuple(map(int,np.concatenate([c['xyz'],width])))
                    masks[cell]=int(c['versions'])
                stats['block_version_runs']-=sum(1 for v in masks.values() for _ in _runs(v,ti))
                stats['distinct_spatial_blocks_over_time']-=len(masks)
            for c in blocks:
                cell=tuple(map(int,c));masks[cell]=masks.get(cell,0)|(1<<ti)
            masks,extra=coalesce_time_coherent_siblings(masks,side,k)
            stats['joint_xyz_time_merge_count']+=extra
            cells=_cell_struct(None,masks)
            data=zlib.compress(cells.tobytes(),level=6);digest=sha256(data).hexdigest()
            dest=self.root/'blobs'/digest
            if not dest.exists():dest.write_bytes(data)
            if key not in before:stats['tiles_with_geometry']+=1
            stats['distinct_spatial_blocks_over_time']+=len(masks)
            stats['block_version_runs']+=sum(1 for v in masks.values() for _ in _runs(v,ti+1))
            updates[key]={'hash':digest,'cells':len(cells)}
            stats['input_voxels_across_sessions']+=len(codes)
            stats['spatial_blocks_across_sessions']+=len(blocks)
            stats['spatial_merge_count']+=len(codes)-len(blocks)
            added_blocks+=len(blocks);changed_tiles+=int(before.get(key,{}).get('hash')!=digest)
        # Updated manifests and metadata are committed only after all blobs exist.
        after={**before,**updates}
        _put_json(self.root/'tiles.json',after)
        self._directory=after
        newmeta=dict(self.meta);newmeta['sessions']=[*sessions,target_session]
        newmeta['stats']=stats
        _put_json(self.root/'metadata.json',newmeta)
        self._meta=newmeta
        self.rebuild_indexes()
        if gc:self.compact_blobs()
        return {'appended':target_session,'new_blocks':added_blocks,'updated_tile_payloads':changed_tiles,
                'index_rebuilt':True,'storage':self.storage()}

    def compact_blobs(self):
        """Remove unreachable old content-addressed geometry blobs.

        Only run offline: a concurrent reader may still hold an older manifest.
        """
        live={x['hash'] for x in self.directory.values()}
        deleted=0;reclaimed=0
        for file in (self.root/'blobs').iterdir():
            if file.name not in live:
                reclaimed+=file.stat().st_size;file.unlink();deleted+=1
        return {'deleted_unreferenced_blobs':deleted,'reclaimed_bytes':reclaimed}

    def rebuild_indexes(self):
        """Real upstream BST-ID tile-time Cell ID sidecar + Morton hash control.

        The geometry blocks themselves are shared and exact. The compact
        packed 4D block mask is common to both search methods. We do not
        claim an optimized 4D trie: BST uses a vectorized cell-range filter.
        """
        tile_keys=sorted(self.directory)
        xyz=np.asarray([_parse_xyz_key(k) for k in tile_keys],dtype=np.uint32).reshape(-1,3)
        tile_code=np.asarray([_morton3(*[int(x) for x in row]) for row in xyz],dtype=np.uint64)
        np.savez_compressed(self.root/'morton_index.npz',xyz=xyz,codes=tile_code,
                            version=np.asarray(INDEX_VERSION))
        Cell=author_module('region_algebra').Cell
        # One official 4D cell per spatial tile over each temporal half-interval
        # presence run. The full per-voxel temporal membership remains in blobs.
        tile_side=self.meta['tile_side'];sp_zoom=21-int(math.log2(tile_side))
        session=self.meta['sessions']
        ids=[];idx=[];bounds=[]
        for row,key in enumerate(tile_keys):
            records=self._read_tile(key)
            active=0
            for mask in records['versions']:active |= int(mask)
            for lo,hi in _runs(active,len(session)):
                ts=_timestamp(session[lo]); te=_timestamp(session[hi]) if hi<len(session) else MAX_TIME
                for tz,tpre in _dyadic_time_cover(ts,te):
                    cell=Cell.from_parts(x=(sp_zoom,int(xyz[row,0])),
                                         y=(sp_zoom,int(xyz[row,1])),
                                         f=(sp_zoom,int(xyz[row,2])), t=(tz,tpre))
                    word=cell.to_id()
                    if Cell.from_id(word)!=cell:raise AssertionError('official 4D round trip failed')
                    ids.append(np.void(word.to_bytes(17,'big')))
                    idx.append(row)
                    a=xyz[row].astype(np.int64)*tile_side
                    bounds.append((int(a[0]),int(a[0]+tile_side),int(a[1]),int(a[1]+tile_side),
                                   int(a[2]),int(a[2]+tile_side),ts,te))
        np.savez_compressed(self.root/'bst_index.npz',xyz=xyz,post_rows=np.asarray(idx,dtype=np.int32),
                            ids=np.asarray(ids,dtype='V17'),boxes=np.asarray(bounds,dtype=np.int64).reshape(-1,8),
                            version=np.asarray(INDEX_VERSION))
        self._index={}
        return self.storage()

    def _read_tile(self,key):
        # Do not cache decoded geometry. Both BST and Morton pay the same
        # decompression+SHA256 cost per tile on every timed query.
        digest=self.directory[key]['hash']
        blob=(self.root/'blobs'/digest).read_bytes()
        if sha256(blob).hexdigest()!=digest:raise IOError('geometry tile checksum mismatch')
        raw=zlib.decompress(blob)
        if len(raw)%BLOCK_DTYPE.itemsize:raise IOError('malformed geometry tile')
        data=np.frombuffer(raw,dtype=BLOCK_DTYPE)
        if len(data)!=self.directory[key]['cells']:raise IOError('geometry record count mismatch')
        return data

    def _load_index(self,mode):
        if mode not in self._index:
            if mode=='morton':
                with np.load(self.root/'morton_index.npz',allow_pickle=False) as f:
                    xyz=f['xyz'];codes=f['codes']
                self._index[mode]={int(k):tuple(map(int,v)) for k,v in zip(codes,xyz)}
            elif mode=='bst':
                with np.load(self.root/'bst_index.npz',allow_pickle=False) as f:
                    boxes=f['boxes'].copy();rows=f['post_rows'].copy();xyz=f['xyz'].copy();ids=f['ids'].copy()
                # Upstream BST-ID identity verified outside the timed path.
                Cell=author_module('region_algebra').Cell
                side=int(self.meta['tile_side']);sz=21-int(math.log2(side))
                for j,r in enumerate(rows):
                    corner=xyz[int(r)]
                    box=boxes[j]
                    # Verify that word's spatial prefixes and time cover are consistent
                    # with the spatial tile.  The exact temporal prefix is recoverable
                    # from the author ID itself.
                    cell=Cell.from_id(int.from_bytes(bytes(ids[j]),'big'))
                    if cell.zooms[:3]!=(sz,sz,sz):
                        raise IOError('stored BST tile cell spatial resolution mismatch')
                    if cell.values[:3]!=tuple(map(int,corner)):
                        raise IOError('stored BST cell spatial prefix mismatch')
                self._index[mode]={'boxes':boxes,'rows':rows,'xyz':xyz}
            else:raise ValueError('index mode should be bst or morton')
        return self._index[mode]

    def _candidate_keys(self,session,bounds,mode):
        meta=self.meta;side=meta['tile_side'];tile_bounds=_tile_range(bounds,meta['source_meta']['dims'],side)
        if tile_bounds is None:return [],0
        ind=self._load_index(mode)
        if mode=='morton':
            selected=[];nprobe=0
            for x in range(tile_bounds[0][0],tile_bounds[0][1]+1):
                for y in range(tile_bounds[1][0],tile_bounds[1][1]+1):
                    for z in range(tile_bounds[2][0],tile_bounds[2][1]+1):
                        nprobe+=1
                        code=_morton3(x,y,z)
                        if code in ind:selected.append(f'{x},{y},{z}')
            return selected,nprobe
        b=ind['boxes'];t=_timestamp(session)
        keep=(b[:,0]<bounds[1])&(b[:,1]>bounds[0])&(b[:,2]<bounds[3])&(b[:,3]>bounds[2])
        keep &= (b[:,4]<bounds[5])&(b[:,5]>bounds[4])&(b[:,6]<=t)&(b[:,7]>t)
        rows=np.unique(ind['rows'][keep]);xyz=ind['xyz']
        return [','.join(map(str,xyz[i])) for i in rows],len(b)

    def query(self,session,bounds,*,mode='bst',stats=False,max_output=10_000_000):
        """Return exact voxel IDs by expanding aggregated *geometry blocks*.

        Block cover only observed cells; clipping at region boundaries is exact.
        Empty/missing observation history is not interpreted as free space.
        """
        t0=time.perf_counter();names=self.meta['sessions']
        if session not in names:raise KeyError(session)
        if len(bounds)!=6 or any(int(bounds[i])>=int(bounds[i+1]) for i in (0,2,4)):
            raise ValueError('invalid half-open voxel query region')
        i=names.index(session)
        q0=time.perf_counter(); keys,probes=self._candidate_keys(session,bounds,mode)
        index_ms=(time.perf_counter()-q0)*1000
        all_xyz=[];blocks_hit=0;total=0
        side=int(self.meta['tile_side']); dims=self.meta['source_meta']['dims']
        spec=self._spec()
        for key in keys:
            tiles=self._read_tile(key)
            active=tiles[(tiles['versions']&(np.uint32(1)<<np.uint32(i)))!=0]
            if not len(active):continue
            base=np.asarray(_parse_xyz_key(key),dtype=np.int64)*side
            origins=base[None,:]+active['xyz'].astype(np.int64)
            width=np.left_shift(1,active['exp'].astype(np.int64))
            volume=np.prod(width,axis=1).astype(np.int64)
            total_candidates=int(np.sum(volume))
            if total_candidates>max_output:raise MemoryError('tile expansion too large')
            row=np.repeat(np.arange(len(active)),volume)
            offset=np.arange(len(row),dtype=np.int64)-np.repeat(np.cumsum(volume)-volume,volume)
            wx=width[row,0];wy=width[row,1];wz=width[row,2]
            xyz=np.empty((len(row),3),dtype=np.int64)
            xyz[:,0]=origins[row,0]+offset//(wy*wz)
            xyz[:,1]=origins[row,1]+(offset//wz)%wy
            xyz[:,2]=origins[row,2]+offset%wz
            keep=np.ones(len(xyz),dtype=bool)
            for j in range(3):
                keep &= (xyz[:,j]>=max(0,int(bounds[2*j]))) & (xyz[:,j]<min(int(dims[j]),int(bounds[2*j+1])))
            xyz=xyz[keep]
            total+=len(xyz)
            if total>max_output:raise MemoryError('expanded geometry ROI exceeds --max-output')
            if len(xyz):
                all_xyz.append(linear_encode(xyz[:,0],xyz[:,1],xyz[:,2],spec))
                blocks_hit+=int(np.sum(np.all((origins+width>np.array(bounds[::2])) & (origins<np.array(bounds[1::2])),axis=1)))
        ans=canonical(np.concatenate(all_xyz) if all_xyz else np.empty(0,dtype=np.uint64))
        if len(ans)!=total:raise IOError('overlapping geometry blocks violate exact occupancy')
        info={'mode':mode,'blocks_read':blocks_hit,'voxels':len(ans),'tile_candidates':len(keys),
              'index_probes':probes,'index_ms':index_ms,'latency_ms':(time.perf_counter()-t0)*1000,
              'replayed_transitions':0}
        return (ans,info) if stats else ans

    def _spec(self):
        from synthetic.world import WorldSpec
        m=self.meta['source_meta'];return WorldSpec(*[int(d)*float(m['voxel_size_m']) for d in m['dims']],float(m['voxel_size_m']))

    def primitives(self,session,bounds,*,mode='bst',stats=False):
        """Return the unexpanded, adaptive geometry blocks overlapping an ROI.

        This is the representation-use path for map geometry applications.
        Exact voxel reconstruction is a deliberately more expensive check.
        A clipped ROI can intersect only PART of a coarse block; callers can
        clip each block box against the ROI for exact occupied-voxel counts.
        """
        clock=time.perf_counter();sessions=self.meta['sessions']
        if session not in sessions:raise KeyError(session)
        if len(bounds)!=6 or any(bounds[j]>=bounds[j+1] for j in (0,2,4)):
            raise ValueError('invalid region')
        q0=time.perf_counter();keys,probes=self._candidate_keys(session,bounds,mode)
        index_ms=(time.perf_counter()-q0)*1000
        outputs=[];side=int(self.meta['tile_side']);bit=np.uint32(1)<<np.uint32(sessions.index(session))
        lower=np.asarray(bounds[::2],dtype=np.int64)
        upper=np.asarray(bounds[1::2],dtype=np.int64)
        for key in keys:
            data=self._read_tile(key)
            active=data[(data['versions']&bit)!=0]
            if not len(active):continue
            start=np.asarray(_parse_xyz_key(key),dtype=np.int64)*side+active['xyz'].astype(np.int64)
            size=1<<active['exp'].astype(np.int64)
            mask=np.all((start+size>lower)&(start<upper),axis=1)
            if np.any(mask):outputs.append(np.column_stack((start[mask],size[mask])).astype(np.int32))
        out=np.concatenate(outputs) if outputs else np.zeros((0,6),dtype=np.int32)
        info={'mode':mode,'primitives':len(out),'index_ms':index_ms,'index_probes':probes,
              'latency_ms':(time.perf_counter()-clock)*1000,'replayed_transitions':0}
        return (out,info) if stats else out

    def describe(self,session,bounds,*,mode='bst'):
        """Return adaptive block geometry for plotting/approximate surfels.

        A cell's centroid and second moments follow analytically from the
        exactly filled regular voxel positions, NOT original laser points.
        """
        if session not in self.meta['sessions']:raise KeyError(session)
        bit=1<<self.meta['sessions'].index(session);side=self.meta['tile_side']
        keys,_=self._candidate_keys(session,bounds,mode)
        output=[];vs=float(self.meta['source_meta']['voxel_size_m'])
        for key in keys:
            base=np.asarray(_parse_xyz_key(key),dtype=np.int64)*side
            for c in self._read_tile(key):
                if not (int(c['versions'])&bit):continue
                xyz=base+c['xyz'].astype(np.int64);w=1<<c['exp'].astype(np.int64)
                if any(xyz[j]>=bounds[2*j+1] or xyz[j]+w[j]<=bounds[2*j] for j in range(3)):continue
                centroid=(xyz+w/2)*vs
                variance=(w.astype(np.float64)**2-1)*(vs**2)/12
                output.append({'min_voxel':xyz.tolist(),'shape_voxels':w.tolist(),
                              'support_count':int(np.prod(w)),
                              'centroid_local_m':centroid.tolist(),
                              'covariance_diagonal_m2':variance.tolist()})
        return output

    def four_d_cells(self,session,bounds,*,limit=1000):
        """Expose true official BST-ID cells over merged geometry & 4D validity.

        This is an export/verification interface, NOT an optimized 4D trie.
        """
        keys,_=self._candidate_keys(session,bounds,'bst');names=self.meta['sessions']
        Cell=author_module('region_algebra').Cell;out=[];side=self.meta['tile_side']
        for key in keys:
            base=np.asarray(_parse_xyz_key(key),dtype=np.int64)*side
            for c in self._read_tile(key):
                w=1<<c['exp'].astype(np.int64);pos=base+c['xyz'].astype(np.int64)
                if any(pos[j]>=bounds[2*j+1] or pos[j]+w[j]<=bounds[2*j] for j in range(3)):continue
                for a,b in _runs(int(c['versions']),len(names)):
                    if not (a<=names.index(session)<b):continue
                    t0=_timestamp(names[a]);t1=_timestamp(names[b]) if b<len(names) else MAX_TIME
                    for z,p in _dyadic_time_cover(t0,t1):
                        parts=[(21-int(c['exp'][j]),int(pos[j])>>int(c['exp'][j])) for j in range(3)]
                        obj=Cell.from_parts(x=parts[0],y=parts[1],f=parts[2],t=(z,p))
                        out.append(obj.to_id().to_bytes(17,'big').hex())
                        if len(out)>=limit:return out
        return out

    def storage(self):
        payload=sum(p.stat().st_size for p in (self.root/'blobs').iterdir())
        manifest=(self.root/'tiles.json').stat().st_size
        meta=(self.root/'metadata.json').stat().st_size
        bst=(self.root/'bst_index.npz').stat().st_size if (self.root/'bst_index.npz').exists() else 0
        morton=(self.root/'morton_index.npz').stat().st_size if (self.root/'morton_index.npz').exists() else 0
        base=payload+manifest+meta
        return {'geometry_payload_bytes':payload,'tile_directory_bytes':manifest,
                'metadata_bytes':meta,'geometry_core_bytes':base,
                'bst_index_bytes':bst,'morton_index_bytes':morton,
                'bst_complete_bytes':base+bst,'morton_complete_bytes':base+morton,
                'stats':self.meta['stats']}
