"""Step 9.7C: reusable surfel eigensystem preprocessing.

Caches *only* quantities derived from frozen map surfels. The local nearest-neighbor
index still uses exactly the same selected centroids as Step 9.7B.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from localization.native_surfel_registration import NativeSurfelConfig, _surfel_normals

@dataclass(frozen=True)
class PreparedSurfelMap:
    normals: np.ndarray
    valid: np.ndarray
    sigmas: np.ndarray
    n: int
    cfg: NativeSurfelConfig

    @classmethod
    def build(cls, cov6, counts, config=None, chunk_size=100000):
        cfg = config or NativeSurfelConfig()
        n=len(counts)
        if len(cov6)!=n: raise ValueError('cov6/counts length mismatch')
        if chunk_size < 1: raise ValueError('chunk_size must be positive')
        normals=np.empty((n,3),dtype=np.float64)
        valid=np.empty(n,dtype=bool)
        sigmas=np.empty(n,dtype=np.float64)
        for lo in range(0,n,chunk_size):
            hi=min(lo+chunk_size,n)
            normals[lo:hi],valid[lo:hi],sigmas[lo:hi]=_surfel_normals(cov6[lo:hi],counts[lo:hi],cfg)
        return cls(normals,valid,sigmas,n,cfg)

    def select(self, indices):
        ids=np.asarray(indices,dtype=np.int64)
        if ids.ndim != 1 or (len(ids) and (ids.min()<0 or ids.max()>=self.n)):
            raise IndexError('local surfel indices outside prepared map')
        return self.normals[ids],self.valid[ids],self.sigmas[ids]
