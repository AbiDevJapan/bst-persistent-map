from __future__ import annotations
from dataclasses import dataclass

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds
from .interval_utils import (
    Signature, bits_required, build_occupied_intervals,
    signature_prefixes, tokens_by_depth, prefix_active,
)


@dataclass(frozen=True)
class _VoxelIntervalRecord:
    voxel: Voxel
    signature_id: int


class BSTIntervalMapPrototype:
    """
    Step-5A ablation: temporal validity prefixes, but NO spatial merging.

    Each voxel stores a temporal occupancy signature represented exactly by
    dyadic time prefixes.  Queries use direct prefix membership at time t,
    eliminating Step-4's 'latest event <= t' predecessor scan.

    Spatial lookup uses a deterministic coarse bucket index.  This class is an
    ablation, not the final BST-ID representation.
    """
    name = 'BSTIntervalMapPrototype'

    def __init__(self, dataset: GeneratedDataset, bucket_voxels: int = 8):
        self.dataset = dataset
        self.bucket_voxels = bucket_voxels
        self.time_bits = bits_required(len(dataset.sessions))
        self.signatures: list[Signature] = []
        self.signature_tokens: list[tuple[tuple[int,int],...]] = []
        self.signature_depth_index: list[dict[int, frozenset[int]]] = []
        self.records: list[_VoxelIntervalRecord] = []
        self._buckets: dict[tuple[int,int,int], list[int]] = {}
        self.bucket_references = 0
        self.last_query_stats = {'blocks_scanned': 0, 'bucket_refs': 0}
        self.last_change_stats = {'blocks_scanned': 0, 'bucket_refs': 0}

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset, bucket_voxels: int = 8):
        obj = cls(dataset, bucket_voxels)
        voxel_intervals = build_occupied_intervals(dataset)
        sig_to_id: dict[Signature, int] = {}

        for v, sig in voxel_intervals.items():
            sid = sig_to_id.get(sig)
            if sid is None:
                sid = len(obj.signatures)
                sig_to_id[sig] = sid
                obj.signatures.append(sig)
                tok = signature_prefixes(sig, len(dataset.sessions))
                obj.signature_tokens.append(tok)
                obj.signature_depth_index.append(tokens_by_depth(tok))
            rid = len(obj.records)
            obj.records.append(_VoxelIntervalRecord(v, sid))
            b = tuple(c // bucket_voxels for c in v)
            obj._buckets.setdefault(b, []).append(rid)
            obj.bucket_references += 1
        return obj

    def _candidate_ids(self, bounds):
        x0,x1,y0,y1,z0,z1 = bounds
        b = self.bucket_voxels
        ids: list[int] = []
        refs = 0
        seen: set[int] = set()
        for bx in range(x0//b, x1//b+1):
            for by in range(y0//b, y1//b+1):
                for bz in range(z0//b, z1//b+1):
                    vals = self._buckets.get((bx,by,bz), ())
                    refs += len(vals)
                    for rid in vals:
                        if rid not in seen:
                            seen.add(rid); ids.append(rid)
        return ids, refs

    def _active(self, sid: int, t: int) -> bool:
        return prefix_active(self.signature_depth_index[sid], t, self.time_bits)

    def region_at_time(self, region: MetricBox, t: int) -> frozenset[Voxel]:
        bounds = metric_box_to_voxel_bounds(self.dataset, region)
        x0,x1,y0,y1,z0,z1 = bounds
        ids, refs = self._candidate_ids(bounds)
        out=set(); scanned=0
        for rid in ids:
            r=self.records[rid]; scanned += 1
            x,y,z=r.voxel
            if x0<=x<=x1 and y0<=y<=y1 and z0<=z<=z1 and self._active(r.signature_id,t):
                out.add(r.voxel)
        self.last_query_stats={'blocks_scanned':scanned,'bucket_refs':refs}
        return frozenset(out)

    def snapshot(self, t: int) -> frozenset[Voxel]:
        s=self.dataset.spec
        return self.region_at_time((0,s.size_x_m,0,s.size_y_m,0,s.size_z_m),t)

    def changes(self, region: MetricBox, t_a: int, t_b: int):
        if t_b < t_a: t_a,t_b=t_b,t_a
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        x0,x1,y0,y1,z0,z1=bounds
        ids,refs=self._candidate_ids(bounds)
        added=set(); removed=set(); scanned=0
        for rid in ids:
            r=self.records[rid]; scanned += 1
            x,y,z=r.voxel
            if not (x0<=x<=x1 and y0<=y<=y1 and z0<=z<=z1): continue
            a=self._active(r.signature_id,t_a); b=self._active(r.signature_id,t_b)
            if a==b: continue
            (added if b else removed).add(r.voxel)
        self.last_change_stats={'blocks_scanned':scanned,'bucket_refs':refs}
        return frozenset(added),frozenset(removed)

    def stats(self):
        return {
            'blocks': len(self.records),
            'unique_temporal_signatures': len(self.signatures),
            'temporal_prefixes': sum(len(x) for x in self.signature_tokens),
            'bucket_references': self.bucket_references,
            'bucket_voxels': self.bucket_voxels,
            'time_bits': self.time_bits,
        }
