"""Step 9.7G: conservative BEV confidence gate. No ground-truth pose information."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class HybridBEVPolicy:
    min_inliers: int = 28
    min_inlier_ratio: float = 0.07
    max_bev_candidates: int = 2

    def __post_init__(self):
        if self.min_inliers < 1 or not (0 < self.min_inlier_ratio <= 1) or self.max_bev_candidates < 1:
            raise ValueError('invalid BEV hybrid policy')

    def accepts(self, result, query_feature_count, reference_feature_count):
        # This heuristic tests feature support, not pose correctness.
        denominator=max(1,min(int(query_feature_count),int(reference_feature_count)))
        return bool(result.converged and np.isfinite(result.T_target_source).all()
                    and result.inliers >= self.min_inliers
                    and result.inliers/denominator >= self.min_inlier_ratio)
