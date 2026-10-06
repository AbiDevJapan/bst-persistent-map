from __future__ import annotations
import math
from representations import (
    SnapshotMap, OctreeTemporalMap, BSTTemporalMapPrototype,
    BSTIntervalMapPrototype, BSTAdaptiveIntervalMapPrototype,
)


def snapshot_compact_bytes(rep: SnapshotMap) -> int:
    return 4 + sum(8 + 6*len(v) for v in rep._sessions.values())


def _octree_node_bytes(node) -> int:
    total=9
    if node.is_leaf: return total+2+3*len(node.times)
    return total+sum(_octree_node_bytes(node.children[i]) for i in sorted(node.children))


def octree_compact_bytes(rep: OctreeTemporalMap) -> int:
    return 8+_octree_node_bytes(rep.root)


def bst_event_bytes(rep: BSTTemporalMapPrototype) -> int:
    key_bytes=math.ceil(rep.encoder.total_bits/8)
    return 16+12*len(rep._buckets)+(key_bytes+1)*rep.event_count


def _temporal_signature_bytes(rep) -> int:
    # signature table: uint16 count; each dyadic temporal prefix uses
    # uint8 depth + ceil(depth/8) bytes prefix value.
    total=4
    for toks in rep.signature_tokens:
        total += 2
        for depth,_ in toks:
            total += 1 + math.ceil(depth/8)
    return total


def bst_interval_bytes(rep: BSTIntervalMapPrototype) -> int:
    # Record = uint16 x,y,z + uint32 signature_id = 10 bytes.
    # Coarse directory = uint32 record-id per reference + 12-byte bucket header.
    header=20
    records=10*len(rep.records)
    directory=12*len(rep._buckets)+4*rep.bucket_references
    return header+records+directory+_temporal_signature_bytes(rep)


def _spatial_prefix_dim_bytes(limit:int,size:int) -> int:
    width=max(1,math.ceil(math.log2(limit)))
    size_bits=int(math.log2(size))
    prefix_len=max(0,width-size_bits)
    return 1 + math.ceil(prefix_len/8)  # depth + prefix bytes


def bst_adaptive_bytes(rep: BSTAdaptiveIntervalMapPrototype) -> int:
    # Each block stores three independent spatial prefixes + uint32 signature id.
    s=rep.dataset.spec
    per_blocks=0
    for b in rep.blocks:
        per_blocks += (
            _spatial_prefix_dim_bytes(s.nx,b.sx)+
            _spatial_prefix_dim_bytes(s.ny,b.sy)+
            _spatial_prefix_dim_bytes(s.nz,b.sz)+4
        )
    directory=12*len(rep._buckets)+4*rep.bucket_references
    return 24+per_blocks+directory+_temporal_signature_bytes(rep)


def compact_storage_bytes(rep) -> int:
    if isinstance(rep,SnapshotMap): return snapshot_compact_bytes(rep)
    if isinstance(rep,OctreeTemporalMap): return octree_compact_bytes(rep)
    if isinstance(rep,BSTTemporalMapPrototype): return bst_event_bytes(rep)
    if isinstance(rep,BSTIntervalMapPrototype): return bst_interval_bytes(rep)
    if isinstance(rep,BSTAdaptiveIntervalMapPrototype): return bst_adaptive_bytes(rep)
    raise TypeError(type(rep))
