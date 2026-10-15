from __future__ import annotations
"""Exact, content-addressed versioned observed-endpoint maps.
Tile blobs are immutable; unchanged tile payloads are shared across versions.
Absence is NEVER called free space. All results use canonical uint64 voxels.
"""
from pathlib import Path
from hashlib import sha256
import io,json,zlib,time
import numpy as np
from realdata.nclt_packed import linear_decode,linear_encode
from .codec import Codec,Frame

SESSIONS=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']
def canonical(a):
    a=np.asarray(a,dtype=np.uint64).reshape(-1)
    return np.unique(a)
def _bytes(a):
    return np.asarray(a,dtype='<u8').tobytes()
def _from_bytes(b):
    return np.frombuffer(b,dtype='<u8').astype(np.uint64,copy=True)
def _tile_key(x,y,z,side):return f'{x//side},{y//side},{z//side}'
def _region_filter(codes,spec,bounds):
    if not len(codes):return codes
    x,y,z=linear_decode(codes,spec)
    x0,x1,y0,y1,z0,z1=bounds
    return codes[(x>=x0)&(x<x1)&(y>=y0)&(y<y1)&(z>=z0)&(z<z1)]
class VersionedStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'blobs').mkdir(exist_ok=True)
        self.meta_file=self.root/'metadata.json'
        self.versions=self.root/'versions.json'
        self._index_cache={}
        self._manifest_cache=None
        self._meta_cache=None
    def create(self,prepared,codec='official',tile_side=32,frame=None):
        if self.meta_file.exists():raise FileExistsError('Database exists; choose another --database')
        if tile_side<=0 or tile_side & (tile_side-1):raise ValueError('tile_side must be power of two')
        import dataclasses
        f=frame or Frame(voxel_size=prepared.spec.voxel_size_m,
                         metric_origin_x=prepared.origin_local_m[0],
                         metric_origin_y=prepared.origin_local_m[1],
                         metric_origin_z=prepared.origin_local_m[2])
        dims=[prepared.spec.nx,prepared.spec.ny,prepared.spec.nz]
        if any(i<=0 for i in dims):raise ValueError('invalid world dimensions')
        metadata={'schema':1,'codec':codec,'tile_side':tile_side,'dims':dims,
                  'voxel_size_m':prepared.spec.voxel_size_m,'frame':dataclasses.asdict(f),
                  'sessions':prepared.session_names,'semantic':'observed LiDAR endpoints only; absence is not free-space evidence',
                  'codec_role':'auxiliary tile identifier; exact filtering performed in local integer voxel space'}
        self.meta_file.write_text(json.dumps(metadata,indent=2))
        self.versions.write_text('{}')
        for name,codes in zip(prepared.session_names,prepared.session_codes):self.ingest(name,codes)
    def _cached_json(self,path,attr,*,clear_indexes=False):
        st=path.stat()
        tag=(st.st_mtime_ns,st.st_ctime_ns,st.st_size)
        cached=getattr(self,attr)
        if cached is not None and cached[0]==tag:return cached[1]
        data=json.loads(path.read_text())
        setattr(self,attr,(tag,data))
        if clear_indexes:self._index_cache.clear()
        return data

    def meta(self):
        return self._cached_json(self.meta_file,'_meta_cache')

    def all_versions(self):
        return self._cached_json(self.versions,'_manifest_cache',clear_indexes=True)
    def _spec(self):
        from synthetic.world import WorldSpec
        m=self.meta(); nx,ny,nz=m['dims']; v=m['voxel_size_m']
        return WorldSpec(nx*v,ny*v,nz*v,v)
    def _blocks(self,codes):
        m=self.meta();s=self._spec();side=m['tile_side'];c=canonical(codes)
        x,y,z=linear_decode(c,s)
        if len(c) and (np.any(x<0)|np.any(x>=s.nx)|np.any(y<0)|np.any(y>=s.ny)|np.any(z<0)|np.any(z>=s.nz)):
            raise ValueError('out-of-bounds voxel code')
        keys=np.stack((x//side,y//side,z//side),axis=1) if len(c) else np.empty((0,3),dtype=int)
        groups={}
        for i,k in enumerate(keys):groups.setdefault(tuple(map(int,k)),[]).append(int(c[i]))
        return {','.join(map(str,k)):canonical(v) for k,v in groups.items()}
    def ingest(self,name,codes):
        versions=self.all_versions();m=self.meta()
        if name in versions:raise ValueError('Session already ingested')
        allowed=m['sessions'];idx=len(versions)
        if idx>=len(allowed) or allowed[idx]!=name:raise ValueError('Chronological ingest required')
        tiles=self._blocks(codes);manifest={}; codec=Codec(m['codec'],Frame(**m['frame']))
        session_time=int(__import__('datetime').datetime.fromisoformat(name).replace(tzinfo=__import__('datetime').timezone.utc).timestamp())
        for key,values in sorted(tiles.items()):
            blob=zlib.compress(_bytes(values),level=6)
            digest=sha256(blob).hexdigest();p=self.root/'blobs'/digest
            if not p.exists():p.write_bytes(blob)
            tx,ty,tz=map(int,key.split(','));side=m['tile_side']
            manifest[key]={'hash':digest,'count':len(values),
                           'id':codec.label(tx*side,ty*side,tz*side,session_time)}
        versions[name]=manifest
        tmp=self.versions.with_suffix('.tmp');tmp.write_text(json.dumps(versions,separators=(',',':')));tmp.replace(self.versions)
        return {'tiles':len(manifest),'voxels':sum(x['count'] for x in manifest.values())}
    def build_spatial_index(self,session,mode='bst_prefix'):
        from .region_index import build_index
        self._index_cache.pop((session,mode),None)
        return build_index(self,session,mode)

    def _matching_keys(self,manifest,bounds,*,session=None,index_mode='scan',with_stats=False):
        if index_mode!='scan':
            from .region_index import INDEX_MODES,load_index
            if index_mode not in INDEX_MODES:raise ValueError(index_mode)
            if session is None:raise ValueError('indexed query requires historical session')
            cache_key=(session,index_mode)
            record=self._index_cache.get(cache_key)
            if record is None:
                record=load_index(self,session,index_mode)
                self._index_cache[cache_key]=record
            matches,stat=record.keys(bounds)
            if any(k not in manifest for k in matches):raise IOError('corrupt spatial index')
            return (matches,stat) if with_stats else matches
        side=self.meta()['tile_side'];x0,x1,y0,y1,z0,z1=bounds
        keys=[k for k in manifest if (lambda a,b,c:(a*side<x1 and (a+1)*side>x0 and b*side<y1 and (b+1)*side>y0 and c*side<z1 and (c+1)*side>z0))(*map(int,k.split(',')))]
        return (keys,{'index_probes':len(manifest),'keys_examined':len(manifest),'prefix_candidates':len(keys)}) if with_stats else keys
    def query(self,session,bounds,*,stats=False,index_mode='scan'):
        start=time.perf_counter()
        versions=self.all_versions();manifest=versions[session];spec=self._spec()
        index_start=time.perf_counter();keys,selection_stats=self._matching_keys(manifest,bounds,session=session,index_mode=index_mode,with_stats=True)
        index_ms=(time.perf_counter()-index_start)*1000
        parts=[];read=0
        for k in keys:
            compressed=(self.root/'blobs'/manifest[k]['hash']).read_bytes();read+=len(compressed)
            if sha256(compressed).hexdigest()!=manifest[k]['hash']:raise IOError('corrupt blob')
            parts.append(_region_filter(_from_bytes(zlib.decompress(compressed)),spec,bounds))
        result=canonical(np.concatenate(parts) if parts else np.empty(0,dtype=np.uint64))
        metrics={'index_mode':index_mode, **selection_stats,'index_ms':index_ms,'tiles_read':len(keys),'compressed_bytes_read':read,
                 'latency_ms':(time.perf_counter()-start)*1000}
        return (result,metrics) if stats else result
    def changes(self,from_session,to_session,bounds,*,index_mode='scan'):
        a=self.query(from_session,bounds,index_mode=index_mode);b=self.query(to_session,bounds,index_mode=index_mode)
        return np.setdiff1d(b,a),np.setdiff1d(a,b)
    def sync_packet(self,source_version,target_version,bounds,*,index_mode='scan'):
        """Transfer source-version region tile payloads absent/mismatched in receiver.
        Receiver must possess the entire source_version manifest and blobs.
        Boundary tiles are shipped whole, and exact ROI is filtered at query time.
        """
        versions=self.all_versions();have=versions[source_version];want=versions[target_version]
        touched=self._matching_keys(want,bounds,session=target_version,index_mode=index_mode)
        # Include removed tiles so receiver can reconstruct a deletion.
        touched=sorted(set(touched)|set(self._matching_keys(have,bounds,session=source_version,index_mode=index_mode)))
        entries={k:want.get(k) for k in touched};blobs={}
        for item in entries.values():
            if item and not any(e.get('hash')==item['hash'] for e in have.values()):
                digest=item['hash'];blobs[digest]=(self.root/'blobs'/digest).read_bytes()
        packet={'source':source_version,'target':target_version,'entries':entries,'blobs':{h:b.hex() for h,b in blobs.items()}}
        return packet
    def apply_packet(self,packet):
        # Called on a clone with the source version present; produce partial region state.
        versions=self.all_versions();source=packet['source'];target=packet['target']
        if source not in versions:raise ValueError('receiver lacks source version')
        if target in versions:raise ValueError('target already exists')
        result=dict(versions[source]);
        for k,item in packet['entries'].items():
            if item is None:result.pop(k,None)
            else:result[k]=item
        for digest,hexblob in packet['blobs'].items():
            blob=bytes.fromhex(hexblob)
            if sha256(blob).hexdigest()!=digest:raise ValueError('corrupt synchronized blob')
            (self.root/'blobs'/digest).write_bytes(blob)
        versions[target]=result
        self.versions.write_text(json.dumps(versions,separators=(',',':')))
        self._index_cache.clear()
    def disk_stats(self):
        blobs=list((self.root/'blobs').iterdir())
        blob_bytes=sum(p.stat().st_size for p in blobs)
        index_files=list((self.root/'indexes').glob('*.npz')) if (self.root/'indexes').exists() else []
        index_bytes=sum(p.stat().st_size for p in index_files)
        manifest_bytes=self.versions.stat().st_size
        metadata_bytes=self.meta_file.stat().st_size
        base=blob_bytes+manifest_bytes+metadata_bytes
        return {'blob_bytes':blob_bytes,'blob_count':len(blobs),
                'manifest_bytes':manifest_bytes,'metadata_bytes':metadata_bytes,
                'index_bytes':index_bytes,'index_file_count':len(index_files),
                'base_database_bytes':base,'total_bytes':base+index_bytes}
