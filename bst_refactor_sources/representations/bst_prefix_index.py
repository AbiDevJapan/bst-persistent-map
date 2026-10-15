from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds
from .interval_utils import dyadic_cover, prefix_active
from .adaptive_block_explicit import AdaptiveBlockExplicitMap
from .bst_adaptive import RectBlock


def _width(limit: int) -> int:
    return max(1, math.ceil(math.log2(limit)))


def _block_token(origin: int, size: int, width: int) -> tuple[int,int]:
    # Blocks from Step 5 are power-of-two sized and aligned to their size.
    exp = int(math.log2(size))
    depth = width - exp
    prefix = origin >> exp
    return depth, prefix


class _AxisPrefixIndex:
    """
    Prefix postings for one spatial axis.

    postings[depth][prefix] -> tuple(block_ids)
    sorted_prefixes[depth]  -> sorted existing prefix values

    Query intervals are exactly dyadically covered. Candidate retrieval finds
    stored block prefixes that are ancestors OR descendants of each query
    prefix, which is equivalent to 1-D overlap for dyadic aligned intervals.
    """

    def __init__(self, width: int):
        self.width = width
        self.postings: dict[int, dict[int, list[int]]] = {}
        self.sorted_prefixes: dict[int, list[int]] = {}
        self.posting_references = 0

    def add(self, depth: int, prefix: int, block_id: int):
        self.postings.setdefault(depth, {}).setdefault(prefix, []).append(block_id)
        self.posting_references += 1

    def finalize(self):
        self.sorted_prefixes = {
            d: sorted(mp.keys()) for d, mp in self.postings.items()
        }

    def query(self, lo: int, hi: int):
        # inclusive [lo,hi]
        tokens = dyadic_cover(lo, hi+1, self.width)
        ids=set()
        posting_lists_touched=0
        posting_refs_touched=0

        for qdepth, qprefix in tokens:
            # Stored prefix is an ancestor of query prefix.
            for d in range(0, qdepth+1):
                mp = self.postings.get(d)
                if not mp:
                    continue
                p = 0 if d == 0 else qprefix >> (qdepth-d)
                vals = mp.get(p)
                if vals:
                    posting_lists_touched += 1
                    posting_refs_touched += len(vals)
                    ids.update(vals)

            # Stored prefix is a strict descendant of query prefix.
            for d in range(qdepth+1, self.width+1):
                mp = self.postings.get(d)
                if not mp:
                    continue
                vals_sorted = self.sorted_prefixes[d]
                shift = d-qdepth
                low = qprefix << shift
                high = ((qprefix+1) << shift) - 1
                i = bisect_left(vals_sorted, low)
                j = bisect_right(vals_sorted, high)
                for p in vals_sorted[i:j]:
                    vals = mp[p]
                    posting_lists_touched += 1
                    posting_refs_touched += len(vals)
                    ids.update(vals)

        return ids, posting_lists_touched, posting_refs_touched


class BSTPrefixIndexedAdaptiveMapPrototype:
    """
    Step-7 prefix-index ablation.

    Uses exactly the SAME RectBlocks and temporal signatures as
    AdaptiveBlockExplicitMap / BSTAdaptiveIntervalMapPrototype, but replaces
    the coarse fixed-size bucket directory with a hierarchical dyadic-prefix
    index along x/y/z.

    This is still a local robotics prototype, NOT a verified implementation of
    the published BST-ID codec. It is designed to answer a narrower question:

      Given identical adaptive blocks, does hierarchical prefix indexing itself
      improve spatial candidate lookup over an ordinary coarse bucket directory?
    """

    name = "BSTPrefixIndexedAdaptiveMapPrototype"

    def __init__(self, dataset: GeneratedDataset):
        self.dataset=dataset
        self.time_bits=1
        self.signatures=[]
        self.signature_tokens=[]
        self.signature_depth_index=[]
        self.blocks: list[RectBlock]=[]
        self.unmerged_voxels=0
        self.xwidth=_width(dataset.spec.nx)
        self.ywidth=_width(dataset.spec.ny)
        self.zwidth=_width(dataset.spec.nz)
        self.xindex=_AxisPrefixIndex(self.xwidth)
        self.yindex=_AxisPrefixIndex(self.ywidth)
        self.zindex=_AxisPrefixIndex(self.zwidth)
        self.last_query_stats={"blocks_scanned":0,"prefix_lists":0,"posting_refs":0,"voxels_emitted":0}
        self.last_change_stats={"blocks_scanned":0,"prefix_lists":0,"posting_refs":0,"voxels_emitted":0}

    @classmethod
    def from_dataset(cls, dataset: GeneratedDataset):
        obj=cls(dataset)

        # Build through the ordinary-ID implementation to GUARANTEE identical
        # blocks/signatures rather than accidentally changing the algorithm.
        base=AdaptiveBlockExplicitMap.from_dataset(dataset)
        obj.time_bits=base.time_bits
        obj.signatures=base.signatures
        obj.signature_tokens=base.signature_tokens
        obj.signature_depth_index=base.signature_depth_index
        obj.blocks=base.blocks
        obj.unmerged_voxels=base.unmerged_voxels

        for rid,b in enumerate(obj.blocks):
            dx,px=_block_token(b.x0,b.sx,obj.xwidth)
            dy,py=_block_token(b.y0,b.sy,obj.ywidth)
            dz,pz=_block_token(b.z0,b.sz,obj.zwidth)
            obj.xindex.add(dx,px,rid)
            obj.yindex.add(dy,py,rid)
            obj.zindex.add(dz,pz,rid)

        obj.xindex.finalize(); obj.yindex.finalize(); obj.zindex.finalize()
        return obj

    def block_fingerprint(self):
        return tuple(self.blocks)

    def _candidate_ids(self,bounds):
        x0,x1,y0,y1,z0,z1=bounds
        xs,xl,xr=self.xindex.query(x0,x1)
        ys,yl,yr=self.yindex.query(y0,y1)
        zs,zl,zr=self.zindex.query(z0,z1)
        # Intersect the smallest sets first.
        sets=sorted((xs,ys,zs), key=len)
        ids = sets[0].intersection(sets[1],sets[2]) if sets else set()
        return ids, xl+yl+zl, xr+yr+zr

    @staticmethod
    def _intersection(b:RectBlock,bounds):
        x0,x1,y0,y1,z0,z1=bounds
        ax0=max(x0,b.x0); ax1=min(x1,b.x0+b.sx-1)
        ay0=max(y0,b.y0); ay1=min(y1,b.y0+b.sy-1)
        az0=max(z0,b.z0); az1=min(z1,b.z0+b.sz-1)
        if ax0>ax1 or ay0>ay1 or az0>az1:return None
        return ax0,ax1,ay0,ay1,az0,az1

    def _active(self,sid,t):
        return prefix_active(self.signature_depth_index[sid],t,self.time_bits)

    @staticmethod
    def _emit(bounds,out:set[Voxel]):
        x0,x1,y0,y1,z0,z1=bounds;n=0
        for x in range(x0,x1+1):
            for y in range(y0,y1+1):
                for z in range(z0,z1+1):
                    out.add((x,y,z));n+=1
        return n

    def region_at_time(self,region:MetricBox,t:int):
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        ids,lists,refs=self._candidate_ids(bounds)
        out=set();scanned=0;emitted=0
        for rid in ids:
            b=self.blocks[rid];scanned+=1
            inter=self._intersection(b,bounds)
            if inter is None or not self._active(b.signature_id,t):continue
            emitted += self._emit(inter,out)
        self.last_query_stats={
            "blocks_scanned":scanned,"prefix_lists":lists,
            "posting_refs":refs,"voxels_emitted":emitted,
        }
        return frozenset(out)

    def snapshot(self,t:int):
        s=self.dataset.spec
        return self.region_at_time((0,s.size_x_m,0,s.size_y_m,0,s.size_z_m),t)

    def changes(self,region:MetricBox,t_a:int,t_b:int):
        if t_b<t_a:t_a,t_b=t_b,t_a
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        ids,lists,refs=self._candidate_ids(bounds)
        added=set();removed=set();scanned=0;emitted=0
        for rid in ids:
            b=self.blocks[rid];scanned+=1
            inter=self._intersection(b,bounds)
            if inter is None:continue
            a=self._active(b.signature_id,t_a);c=self._active(b.signature_id,t_b)
            if a==c:continue
            emitted += self._emit(inter,added if c else removed)
        self.last_change_stats={
            "blocks_scanned":scanned,"prefix_lists":lists,
            "posting_refs":refs,"voxels_emitted":emitted,
        }
        return frozenset(added),frozenset(removed)

    def index_stats(self):
        def one(idx):
            distinct=sum(len(mp) for mp in idx.postings.values())
            return {"distinct_prefixes":distinct,"posting_references":idx.posting_references}
        return {"x":one(self.xindex),"y":one(self.yindex),"z":one(self.zindex)}

    def stats(self):
        ix=self.index_stats()
        return {
            "blocks":len(self.blocks),
            "unmerged_voxels":self.unmerged_voxels,
            "spatial_compression_ratio":len(self.blocks)/self.unmerged_voxels if self.unmerged_voxels else 0.0,
            "unique_temporal_signatures":len(self.signatures),
            "temporal_prefixes":sum(len(x) for x in self.signature_tokens),
            "x_prefixes":ix["x"]["distinct_prefixes"],
            "y_prefixes":ix["y"]["distinct_prefixes"],
            "z_prefixes":ix["z"]["distinct_prefixes"],
            "prefix_posting_references":sum(ix[a]["posting_references"] for a in ("x","y","z")),
            "time_bits":self.time_bits,
        }
