# BST lifelong mapping prototype — through Step 9.3.1

This archive contains the cumulative research prototype through **Step 9.3.1: history-pattern-aware robust localization**.

The spatial representation remains frozen at the Step-9.2 compact balanced K=4 localization LOD over the Step-8.5 lifelong BST/history map. Step 9.3.1 compares the frozen unweighted matcher, the Step-9.3 linear repeated-occupancy weighting control, and a parameter-free residual-gated weighting based on causal prior-session occupancy-transition patterns.

See `README_STEP9_3_1.md` for the experiment contract and recommended held-out command.

`PrototypeBSTSpatialCodec` remains a local research codec, not a verified implementation of an upstream BST-ID specification.
