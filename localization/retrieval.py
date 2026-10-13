"""Interchangeable frozen historical scan retrievers; GT is never consulted."""
from __future__ import annotations
import numpy as np
from localization.bevplace import BEVPlaceDescriptorCache
from localization.global_frontend import ScanContextDatabase, make_scan_context
from localization.pyramid import voxel_downsample_first

class HistoricalRetriever:
    def __init__(self, kind, *, cache_dir=None, sc_config=None, top_k=5, candidate_pool=30, descriptor_voxel_m=0.5):
        if kind not in {'bevplace', 'scan_context'}:
            raise ValueError('retriever must be bevplace or scan_context')
        self.kind = kind
        if kind == 'bevplace' and not cache_dir:
            raise ValueError('BEVPlace++ requires cached, pretrained descriptors')
        self.descriptor_cache = BEVPlaceDescriptorCache(cache_dir) if kind == 'bevplace' else None
        self.sc_config = sc_config
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.descriptor_voxel_m = float(descriptor_voxel_m)
        if self.top_k < 1:
            raise ValueError('top_k must be >= 1')
        self._pairs = {}
        self._sc_db = None

    def load_reference(self, prior_session, target_sessions, ref):
        if self.kind == 'bevplace':
            self._pairs = {}
            for target_session in target_sessions:
                db, pair = self.descriptor_cache.load_pair(
                    prior_session, target_session, ref['scan_indices'],
                    [ref['scans'][int(i)].name for i in ref['scan_indices']])
                self._pairs[target_session] = (db, pair)
            self._sc_db = None
            return int(next(iter(self._pairs.values()))[0].descriptors.nbytes)
        self._sc_db = ScanContextDatabase(ref['descriptors'], ref['ring_keys'])
        self._pairs = {}
        return 0

    def query(self, query_session, scan_filename, original_scan_index, body):
        if self.kind == 'bevplace':
            db, pair = self._pairs[query_session]
            descriptor = self.descriptor_cache.query_descriptor(pair['query_by_name'],scan_filename,int(original_scan_index))
            return db.query_descriptor(descriptor, top_k=self.top_k)
        desc = make_scan_context(voxel_downsample_first(body,self.descriptor_voxel_m),self.sc_config)
        return self._sc_db.query_descriptor(desc,top_k=self.top_k,candidate_pool=self.candidate_pool)

    @property
    def provenance(self):
        return next(iter(self._pairs.values()))[1]['provenance'] if self._pairs else None
