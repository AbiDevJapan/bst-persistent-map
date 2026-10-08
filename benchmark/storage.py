from __future__ import annotations
import math
from representations import (
    SnapshotMap, OctreeTemporalMap, BSTTemporalMapPrototype,
    BSTIntervalMapPrototype, BSTAdaptiveIntervalMapPrototype,
    OctreeIntervalMap, OctreeSharedSignatureMap,
    AdaptiveBlockExplicitMap, BSTPrefixIndexedAdaptiveMapPrototype,
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



def _temporal_token_bytes(tokens) -> int:
    return sum(1 + math.ceil(depth/8) for depth,_ in tokens)


def _octree_interval_node_bytes(node) -> int:
    total=9
    if node.is_leaf:
        return total + 2 + _temporal_token_bytes(node.tokens)
    return total + sum(_octree_interval_node_bytes(node.children[i]) for i in sorted(node.children))


def octree_interval_bytes(rep: OctreeIntervalMap) -> int:
    return 8 + _octree_interval_node_bytes(rep.root)


def _octree_shared_node_bytes(node) -> int:
    total=9
    if node.is_leaf:
        return total + 4  # uint32 signature id
    return total + sum(_octree_shared_node_bytes(node.children[i]) for i in sorted(node.children))


def octree_shared_bytes(rep: OctreeSharedSignatureMap) -> int:
    return 12 + _octree_shared_node_bytes(rep.root) + _temporal_signature_bytes(rep)


def adaptive_explicit_bytes(rep: AdaptiveBlockExplicitMap) -> int:
    """
    Ordinary-ID identical-block baseline.

    Block record:
      uint16 x0,y0,z0,sx,sy,sz = 12 bytes
      uint32 signature_id       = 4 bytes
      total                     = 16 bytes

    Same coarse bucket directory as bst_adaptive.
    """
    header=24
    records=16*len(rep.blocks)
    directory=12*len(rep._buckets)+4*rep.bucket_references
    return header+records+directory+_temporal_signature_bytes(rep)


def _axis_prefix_index_bytes(axis_index) -> int:
    """
    Logical compact postings layout:
      uint16 depth_group_count
      per depth: uint8 depth + uint32 prefix_count
      per prefix: packed prefix + uint32 posting_count + uint32 block_ids[]
    """
    total=2
    for depth, mp in axis_index.postings.items():
        total += 1 + 4
        pbytes=math.ceil(depth/8)
        for prefix, ids in mp.items():
            total += pbytes + 4 + 4*len(ids)
    return total


def bst_prefix_index_bytes(rep: BSTPrefixIndexedAdaptiveMapPrototype) -> int:
    # Same compact prefix-coded blocks/signatures as bst_adaptive, but with
    # 3 independent hierarchical prefix-posting indexes instead of buckets.
    s=rep.dataset.spec
    block_bytes=0
    for b in rep.blocks:
        block_bytes += (
            _spatial_prefix_dim_bytes(s.nx,b.sx)+
            _spatial_prefix_dim_bytes(s.ny,b.sy)+
            _spatial_prefix_dim_bytes(s.nz,b.sz)+4
        )
    index_bytes=(
        _axis_prefix_index_bytes(rep.xindex)+
        _axis_prefix_index_bytes(rep.yindex)+
        _axis_prefix_index_bytes(rep.zindex)
    )
    return 24+block_bytes+index_bytes+_temporal_signature_bytes(rep)

def compact_storage_bytes(rep) -> int:
    if isinstance(rep,SnapshotMap): return snapshot_compact_bytes(rep)
    if isinstance(rep,OctreeTemporalMap): return octree_compact_bytes(rep)
    if isinstance(rep,OctreeIntervalMap): return octree_interval_bytes(rep)
    if isinstance(rep,OctreeSharedSignatureMap): return octree_shared_bytes(rep)
    if isinstance(rep,BSTTemporalMapPrototype): return bst_event_bytes(rep)
    if isinstance(rep,BSTIntervalMapPrototype): return bst_interval_bytes(rep)
    if isinstance(rep,BSTAdaptiveIntervalMapPrototype): return bst_adaptive_bytes(rep)
    if isinstance(rep,AdaptiveBlockExplicitMap): return adaptive_explicit_bytes(rep)
    if isinstance(rep,BSTPrefixIndexedAdaptiveMapPrototype): return bst_prefix_index_bytes(rep)
    raise TypeError(type(rep))
