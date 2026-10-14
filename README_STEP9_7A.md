# Step 9.7A — Orientation and refinement diagnostics (NO optimization changes)

This patch is for the **unified/refactored** localization repository from `bst_localization_refactor.zip`, **not** the historical Step 9.6 runner. Apply as a separate Git commit. It deliberately does not change retrieval, KISS-Matcher, small_gicp parameters, map geometry, success thresholds, or `T_final` selection.

## First: analyze Step 9.6 results without rerunning NCLT

From repository root:

```bash
python run_step9_7a.py \
  --results results/step9_6_bevplace_global_to_local \
  --output results/step9_7a_orientation
```

This works on the already-uploaded Step 9.6 `localization_trials.csv` and `frontend_trials.csv`; no raw NCLT, descriptor extraction, network inference, or KISS-Matcher is required. Produces:

- `orientation_trials.csv`: 400 rows (200 queries × voxel/BST), original metrics plus pipeline failure labels, independent position/yaw thresholds and before/after fine transitions.
- `orientation_summary.csv`: pooled and session-level decomposition.
- `failure_categories.csv`: conditional hierarchical categories to prioritize investigation. These are not causal proof.
- `orientation_metadata.json`: explicitly labels which metrics require an instrumented rerun.

The old Step 9.6 output stores **absolute** yaw errors only, so no honest computation of the signed error, signed update, or yaw observability is possible from those CSVs alone.

## Second: optional NCLT rerun for signed yaw and geometry probes

Only after your **behavior-preserving unified refactor passes** its golden-checker:

```bash
python run_localization.py \
  --retriever bevplace \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 \
  --top-k 5 \
  --orientation-diagnostics \
  --output results/step9_7a_instrumented

python check_localization_regression.py \
  --candidate results/step9_7a_instrumented

python run_step9_7a.py \
  --results results/step9_7a_instrumented \
  --output results/step9_7a_instrumented_analysis
```

The `--orientation-diagnostics` switch saves a **separate** `orientation_pose_audit.csv`. It logs:

- signed frontend/final yaw relative to GT (evaluation only); signed map yaw update;
- signed yaw magnitude improvement, translation update magnitude;
- GICP converged flag, raw GICP optimizer `e`, iteration and inlier count, and failure reason;
- optional nearest-neighbor overlap fraction and inlier RMSE on the **actual selected pose**;
- optional finite-difference curvature of an independent truncated point-to-point NN cost at ±0.5° yaw (all other pose variables held fixed).

**IMPORTANT:** The yaw curvature is a *diagnostic surrogate*, **not** the 6×6 small_gicp Hessian, a calibrated posterior covariance, or a proof of yaw observability. It is derived from the local target centroids and scan points. Do not compare its raw magnitude across voxel/BST as though objective scales were equivalent. SmallGICP Python's `e` is also stored without assuming an identical normalization across maps. SciPy enables the NN probe; without it, its values are blank, while all other diagnostics still work.

Instrumented probes run after `total_ms` is recorded, so diagnostic compute time is **not** included in the benchmark's reported registration latency. For production latency, account for diagnostic time separately. GT is never used for pose selection or candidate ranking.

## Frozen reference counts expected

Existing completed Step 9.6 / 200 NCLT queries, top 5 BEVPlace++:

| Metric | BST K=4 | Voxel |
|---|---:|---:|
| Fine XY ≤0.5 m AND yaw ≤1° | 95 | 103 |
| XY ≤0.5 m regardless of yaw | 158 | 162 |
| Yaw ≤1° regardless of XY | 101 | 104 |
| XY succeeds, yaw fails | 63 | 59 |
| Frontend fine successes | 96 | 96 |
| Frontend fine → final not fine | 12 | 11 |
| Frontend not fine → final fine | 11 | 18 |
| Local GICP converged | 88 | 106 |
| Final fine with non-converged GICP (frontend fallback) | 47 | 33 |

A report that disagrees with these counts should not be used for hypothesis testing until the query list, thresholds and input CSVs are reconciled.

## What to conclude from 9.7A

1. **Yaw-limited vs translation-limited**: use the 2×2 threshold decomposition. The 63 BST yaw-only failures are candidates for improvement, but are *not* guaranteed to be correctable by rotation-only changes.
2. **Optimizer drift**: inspect signed and magnitude changes; distinguish convergence from accuracy. Investigate cases where the original frontend is fine but the map pushes it outside the threshold.
3. **Geometry ambiguity**: examine yaw-cost-surrogate profiles, overlap and GICP status across fine successes/failures, controlling for map representation and session.
4. **Retrieval failures**: `no_top5_near_reference` means none of the *retrieved* candidates was within 5 m; it does **not** distinguish absent database coverage from a ranking miss.
5. **NEXT experiment (9.7B)**: choose one intervention based on the measured mechanism and evaluate on held-out queries. Do not tune BST K=4 here.

## Tests

```bash
python -m pytest -q tests/test_orientation_diagnostics.py
python -m pytest -q
```

On the assembled supplied repository sources, the full test suite passes. The actual 200-query instrumented NCLT rerun cannot be executed in the hosted environment because raw NCLT scan/map caches are not available.
