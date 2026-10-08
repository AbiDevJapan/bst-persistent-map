from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np

@dataclass(frozen=True)
class AxisPrefix:
    length:int
    value:int

@dataclass(frozen=True)
class BSTBlockCode:
    x:AxisPrefix
    y:AxisPrefix
    z:AxisPrefix

class SpatialBlockCodec(Protocol):
    """Minimal replaceable interface for a BST-ID spatial codec.

    A future verified upstream BST-ID implementation should implement this
    interface.  Adaptive map construction/query code does not need to change.
    """
    name:str
    def encode_block(self, block, spec) -> BSTBlockCode: ...
    def spatial_table_bytes(self, blocks:np.ndarray, spec) -> int: ...


def _width(n:int): return max(1,math.ceil(math.log2(max(1,n))))

class PrototypeBSTSpatialCodec:
    """Local dyadic-prefix prototype used in Steps 7-8.3.

    NOT a verified implementation of the published/upstream BST-ID codec.
    It exists behind a narrow interface so replacing it later is localized to
    this module (or another implementation of ``SpatialBlockCodec``).
    """
    name='prototype_bst_prefix'

    @staticmethod
    def _axis(origin:int,size:int,dim:int)->AxisPrefix:
        if size<1 or size&(size-1): raise ValueError('adaptive block size must be power-of-two')
        if origin%size: raise ValueError('adaptive block origin must be aligned to size')
        width=_width(dim); exp=int(math.log2(size)); plen=max(0,width-exp)
        return AxisPrefix(plen, int(origin>>exp) if plen else 0)

    def encode_block(self,block,spec)->BSTBlockCode:
        return BSTBlockCode(
            self._axis(int(block[0]),int(block[3]),spec.nx),
            self._axis(int(block[1]),int(block[4]),spec.ny),
            self._axis(int(block[2]),int(block[5]),spec.nz),
        )

    def spatial_table_bytes(self,blocks:np.ndarray,spec)->int:
        if not len(blocks): return 0
        total=np.zeros(len(blocks),dtype=np.int64)
        for dim,sizecol in ((spec.nx,3),(spec.ny,4),(spec.nz,5)):
            width=_width(dim)
            ex=np.log2(blocks[:,sizecol]).astype(np.int64)
            plen=np.maximum(0,width-ex)
            # 1 byte stores prefix length; remaining bytes store prefix bits.
            total += 1 + ((plen+7)//8)
        return int(total.sum())
