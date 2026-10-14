# Step 9.7B — Native-surfel XY+yaw local registration

This **patch** applies on top of the unified localization refactor **and** Step 9.7A. Install in a clean Git branch from your repository root. It changes only the final scan-to-map stage; it does not change BEVPlace++, KISS, query selection, the frozen BST map, or map geometry.

## Install

```bash
unzip -o Step9_7B_Native_Surfel_Registration.zip
python -m pytest -q tests/test_native_surfel_registration.py tests/test_localization_refactor.py tests/test_orientation_diagnostics.py
```

## Supported modes

* `--local-solver gicp` (default): original small_gicp map registration; no changes by default.
* `--local-solver native_surfel`: robust point-to-plane XY+yaw solver using local surfel centroids, `cov6`, and counts; roll, pitch and z stay at the frontend estimate.
* `--native-yaw-multistart`: optional fixed 0, ±1.5-degree starts; more expensive. Not recommended until single-start is tested.
* `--save-frontend-poses`: writes an exact keyed `frontend_poses.npz` file containing each query's selected 4x4 frontend pose.
* `--frontend-pose-cache PATH`: replays those transforms. In replay mode KISS is bypassed, so **the reported total_ms is not a real online latency** and displayed selected-reference diagnostics are based on the retrieval first rank, not the previously selected candidate. Compare final pose accuracies and local solver times, not frontend metrics.

## First produce baseline poses (this run pays the usual KISS cost)

```bash
python run_localization.py --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --local-solver gicp --save-frontend-poses \
  --output results/step9_7b_baseline
```

## Paired comparisons: identical frontend poses

```bash
COMMON="--retriever bevplace --data-root data --query-scans-csv results/step9_5_global_to_local/query_scans.csv --descriptor-cache-dir results/step9_6_bevplace_descriptors --cache-dir .cache/step8_1 --top-k 5 --frontend-pose-cache results/step9_7b_baseline/frontend_poses.npz"

python run_localization.py $COMMON --local-solver gicp --output results/step9_7b_gicp_replay
python run_localization.py $COMMON --local-solver native_surfel --output results/step9_7b_native_single
python run_localization.py $COMMON --local-solver native_surfel --native-yaw-multistart --output results/step9_7b_native_multistart
```

## Evaluation and guardrails

Compare `localization_trials.csv` on the **same query identities**, success, yaw and 6DoF thresholds, per-map failures, and `local_gicp_ms` (legacy CSV column name, now contains the chosen local solver runtime). The summary's `median_local_gicp_ms` and calculated `bst_local_gicp_speedup` retain legacy names. Record local map retrieval separately.

This is an **experimental solver**: a robust nearest-centroid correspondence approximation using native covariance PCA normals; `cov6` influences PCA normal estimation and bounded robust scale, counts gate low-support surfels. It is not probabilistically calibrated full GICP and no success improvement is guaranteed. It currently builds a local KD-tree / performs covariance eigendecomposition **per query**; runtime may exceed small_gicp. Treat speed claims as unproven until timed on NCLT. The solver accepts only geometric-descent solutions inside a bounded XY/yaw update region; when rejected, it returns the exact frontend pose. No ground truth is used during matching.

This patch does not resolve nondeterministic KISS output by itself; it makes the **local optimizer comparison** paired by replaying cached frontend poses. The initial baseline capture can still differ from earlier runs.

## Important artifact caveats

The map still uses `PrototypeBSTSpatialCodec`, not author-artifact BST-ID. The benchmark excludes online BEV image generation and network extraction from total latency. This patch should be validated on separate held-out queries after development, since existing NCLT samples have already guided design choices.
