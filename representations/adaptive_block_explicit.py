from __future__ import annotations

from collections import defaultdict

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds
from .interval_utils import (
    Signature, bits_required, build_occupied_intervals,
    signature_prefixes, tokens_by_depth, prefix_active,
)
from .bst_adaptive import RectBlock, merge_signature_voxels


class AdaptiveBlockExplicitMap:
    """
    Step-7 ordinary-ID control.

    IMPORTANT: This creates EXACTLY the same adaptive RectBlock decomposition
    and temporal signatures as BSTAdaptiveIntervalMapPrototype.

    The only representation difference is that spatial blocks are treated as
    ordinary explicit bounds/IDs rather than compact BST-style prefixes.

    Runtime indexing deliberately uses the SAME coarse spatial bucket directory
    as the Step-5/6 bst_adaptive implementation. Therefore:

      adaptive_explicit vs bst_adaptive

    isolates spatial-record encoding/storage. Query latency should be nearly
    identical; if it is, Step-6 query speed came from the adaptive blocks and
    bucket pruning, not from BST-ID lookup itself.
    """

    name = "AdaptiveBlockExplicitMap"

    def __init__(self, dataset: GeneratedDataset, bucket_voxels: int = 8):
        self.dataset = dataset
        self.bucket_voxels = bucket_voxels
        self.time_bits = bits_required(len(dataset.sessions))
        self.signatures: list[Signature] = []
        self.signature_tokens = []
        self.signature_depth_index = []
        self.blocks: list[RectBlock] = []
        self._buckets: dict[tuple[int,int,int], list[int]] = {}
        self.bucket_references = 0
        self.unmerged_voxels = 0
        self.last_query_stats = {"blocks_scanned": 0, "bucket_refs": 0, "voxels_emitted": 0}
        self.last_change_stats = {"blocks_scanned": 0, "bucket_refs": 0, "voxels_emitted": 0}

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset, bucket_voxels: int = 8):
        obj = cls(dataset, bucket_voxels)

        voxel_intervals = build_occupied_intervals(dataset)
        obj.unmerged_voxels = len(voxel_intervals)
        grouped: dict[Signature, set[Voxel]] = defaultdict(set)
        for v, sig in voxel_intervals.items():
            grouped[sig].add(v)

        limits = (dataset.spec.nx, dataset.spec.ny, dataset.spec.nz)
        for sig, voxels in grouped.items():
            sid = len(obj.signatures)
            obj.signatures.append(sig)
            tok = signature_prefixes(sig, len(dataset.sessions))
            obj.signature_tokens.append(tok)
            obj.signature_depth_index.append(tokens_by_depth(tok))
            obj.blocks.extend(sorted(merge_signature_voxels(voxels, sid, limits)))

        bsz = bucket_voxels
        for rid, b in enumerate(obj.blocks):
            for bx in range(b.x0 // bsz, (b.x0+b.sx-1)//bsz + 1):
                for by in range(b.y0 // bsz, (b.y0+b.sy-1)//bsz + 1):
                    for bz in range(b.z0 // bsz, (b.z0+b.sz-1)//bsz + 1):
                        obj._buckets.setdefault((bx,by,bz), []).append(rid)
                        obj.bucket_references += 1
        return obj

    def block_fingerprint(self):
        return tuple(self.blocks)

    def _candidate_ids(self, bounds):
        x0,x1,y0,y1,z0,z1 = bounds
        b = self.bucket_voxels
        seen=set(); ids=[]; refs=0
        for bx in range(x0//b, x1//b+1):
            for by in range(y0//b, y1//b+1):
                for bz in range(z0//b, z1//b+1):
                    vals = self._buckets.get((bx,by,bz), ())
                    refs += len(vals)
                    for rid in vals:
                        if rid not in seen:
                            seen.add(rid); ids.append(rid)
        return ids, refs

    @staticmethod
    def _intersection(b: RectBlock, bounds):
        x0,x1,y0,y1,z0,z1 = bounds
        ax0=max(x0,b.x0); ax1=min(x1,b.x0+b.sx-1)
        ay0=max(y0,b.y0); ay1=min(y1,b.y0+b.sy-1)
        az0=max(z0,b.z0); az1=min(z1,b.z0+b.sz-1)
        if ax0>ax1 or ay0>ay1 or az0>az1:
            return None
        return ax0,ax1,ay0,ay1,az0,az1

    def _active(self, sid, t):
        return prefix_active(self.signature_depth_index[sid], t, self.time_bits)

    @staticmethod
    def _emit(bounds, out: set[Voxel]):
        x0,x1,y0,y1,z0,z1 = bounds
        n=0
        for x in range(x0,x1+1):
            for y in range(y0,y1+1):
                for z in range(z0,z1+1):
                    out.add((x,y,z)); n += 1
        return n

    def region_at_time(self, region: MetricBox, t: int):
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        ids, refs = self._candidate_ids(bounds)
        out=set(); scanned=0; emitted=0
        for rid in ids:
            b=self.blocks[rid]; scanned += 1
            inter=self._intersection(b,bounds)
            if inter is None or not self._active(b.signature_id,t):
                continue
            emitted += self._emit(inter,out)
        self.last_query_stats = {
            "blocks_scanned": scanned,
            "bucket_refs": refs,
            "voxels_emitted": emitted,
        }
        return frozenset(out)

    def snapshot(self, t: int):
        s=self.dataset.spec
        return self.region_at_time((0,s.size_x_m,0,s.size_y_m,0,s.size_z_m), t)

    def changes(self, region: MetricBox, t_a: int, t_b: int):
        if t_b<t_a:
            t_a,t_b=t_b,t_a
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        ids,refs=self._candidate_ids(bounds)
        added=set(); removed=set(); scanned=0; emitted=0
        for rid in ids:
            b=self.blocks[rid]; scanned += 1
            inter=self._intersection(b,bounds)
            if inter is None:
                continue
            a=self._active(b.signature_id,t_a)
            c=self._active(b.signature_id,t_b)
            if a==c:
                continue
            emitted += self._emit(inter, added if c else removed)
        self.last_change_stats = {
            "blocks_scanned": scanned,
            "bucket_refs": refs,
            "voxels_emitted": emitted,
        }
        return frozenset(added), frozenset(removed)

    def stats(self):
        return {
            "blocks": len(self.blocks),
            "unmerged_voxels": self.unmerged_voxels,
            "spatial_compression_ratio": len(self.blocks)/self.unmerged_voxels if self.unmerged_voxels else 0.0,
            "unique_temporal_signatures": len(self.signatures),
            "temporal_prefixes": sum(len(x) for x in self.signature_tokens),
            "bucket_references": self.bucket_references,
            "bucket_voxels": self.bucket_voxels,
            "time_bits": self.time_bits,
        }
