import sys,tempfile,shutil
from pathlib import Path
import numpy as np
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.codec import Codec

class Prepared:
    def __init__(self):
        self.spec=WorldSpec(64.,64.,16.,1.)
        self.origin_local_m=(0.,0.,0.)
        self.session_names=['2012-01-15','2012-02-02','2012-03-17']
        self.session_codes=[self.code([(1,1,1),(32,32,2),(40,40,2)]),
                            self.code([(1,1,1),(32,32,2),(41,40,2)]),
                            self.code([(1,1,1),(42,40,2)])]
    def code(self,points):
        a=np.array(points,dtype=np.uint64)
        return np.sort(linear_encode(a[:,0],a[:,1],a[:,2],self.spec))

def test_codec_switch_and_query():
    prep=Prepared();b=(0,43,0,43,0,8)
    for kind in ['official','prototype','morton']:
        with tempfile.TemporaryDirectory() as td:
            store=VersionedStore(td);store.create(prep,kind,16)
            for name,codes in zip(prep.session_names,prep.session_codes):
                assert np.array_equal(store.query(name,b),codes)
            added,removed=store.changes(prep.session_names[0],prep.session_names[1],b)
            assert len(added)==len(removed)==1
            assert store.disk_stats()['blob_bytes']>0

def test_region_boundary_and_sync():
    prep=Prepared();b=(32,48,32,48,0,8)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as receiver_dir:
        store=VersionedStore(td);store.create(prep,'official',16)
        full=store.query(prep.session_names[1],b)
        assert len(full)==2
        packet=store.sync_packet(prep.session_names[0],prep.session_names[1],b)
        receiver=VersionedStore(receiver_dir)
        shutil.copyfile(store.meta_file,receiver.meta_file)
        base={prep.session_names[0]:store.all_versions()[prep.session_names[0]]}
        receiver.versions.write_text(__import__('json').dumps(base))
        for entry in base[prep.session_names[0]].values():
            src=store.root/'blobs'/entry['hash'];shutil.copyfile(src,receiver.root/'blobs'/entry['hash'])
        receiver.apply_packet(packet)
        assert np.array_equal(receiver.query(prep.session_names[1],b),full)

def test_official_is_actual_upstream_package():
    codec=Codec('official')
    a=codec.label(0,0,0,1326585600)
    assert a.startswith('official:')
    assert a!=codec.label(32,0,0,1326585600)

def test_corruption_detected():
    prep=Prepared()
    with tempfile.TemporaryDirectory() as td:
        s=VersionedStore(td);s.create(prep,'prototype',16)
        entry=next(iter(s.all_versions()[prep.session_names[0]].values()))
        (s.root/'blobs'/entry['hash']).write_bytes(b'corrupt')
        import pytest
        with pytest.raises(IOError):s.query(prep.session_names[0],(0,64,0,64,0,16))
