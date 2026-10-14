# Step 9.7C — Reusable native-surfel preparation

This is an **implementation-only** speed optimization of Step 9.7B. It does **not** change the solver loss, correspondence search, accepting/rejecting updates, output poses, BEVPlace++ retrieval, KISS, causal-map plan, or BST K=4 representation.

## What changed

`localization/native_surfel_prepared.py` computes covariance eigenvectors, horizontal normal validity, and smallest-eigenvalue standard deviations for each **historical map once**, in bounded chunks. `benchmark/localization_benchmark.py` creates one prepared map per enabled representation, selects prepared data with the exact local `ids`, and calls `align_native_surfels(..., prepared=...)`. The unchanged per-query KD-tree is built from the **same local centroid array** and the original optimization logic is applied.

**Cost**: one-time preparation of several million surfels per historical map (may take time and increase peak RSS). Timing should be reported separately as map initialization; not included in local registration query time. The prepared arrays use about **33 bytes per surfel** (three float64 normals + one bool + one float64 sigma), or ~80 MiB for a 2.48M-surfels BST history map, plus array and allocator overhead. This is a runtime speed-vs-memory tradeoff, and is NOT an improvement in BST serialized-storage compression. Precomputing can be selectively disabled by reverting the benchmark integration; the solver's `prepared` argument is optional.

Note: repeated covariance eigendecomposition is removed. Building a fresh local `cKDTree` each query and numerical iterations remain. Therefore reaching <50 ms is **not guaranteed**. If runtime remains above target, next profile KD-tree build and iteration correspondence search and implement reusable spatial tiles, while checking exact correspondence equivalence.

## Installation

Extract the patch at project root (Step 9.7B must already be installed):

```sh
unzip -o Step9_7C_Prepared_Surfels.zip
python -m pytest -q tests/test_native_surfel_registration.py tests/test_native_surfel_prepared.py tests/test_localization_refactor.py tests/test_orientation_diagnostics.py
```

## Controlled evaluation with the same front-end pose cache

Run the *original Step 9.7B native* code/results as your accuracy and local runtime baseline. Then with 9.7C installed:

```sh
python run_localization.py \
  --retriever bevplace \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 \
  --top-k 5 \
  --frontend-pose-cache results/step9_7b_baseline/frontend_poses.npz \
  --local-solver native_surfel \
  --output results/step9_7c_prepared
```

Expected **algorithmic** invariants: same pose estimates, accepted/rejected convergence statuses, scores, local primitives, and success/failure counts as Step 9.7B native, subject to floating-point differences. Only local solver runtime, session initialization, and peak memory should differ. The cached run excludes the original frontend time; do **not** label its `total_ms` as online end-to-end latency.

Key comparison metrics: median/p95 `local_gicp_ms`, build time for each `PreparedSurfelMap`, peak RSS, voxel/BST fine and strict-6DoF rates, and paired query-level pose/acceptance equivalence. This patch does not instrument one-time preparation duration; measure it separately with the process profiler or an external timer if essential.

Do not change native solver thresholds or activate `--native-yaw-multistart` in one run and not the other.
