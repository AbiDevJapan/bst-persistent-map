# Step 9.7D — Profiled adaptive KISS frontend (first phase)

This is an **incremental and testable first phase**, NOT yet a BEVPlace++ local-feature pose-estimator implementation.

## What changes
- `--frontend-strategy exhaustive` (default) retains frozen top-K registration and the original candidate selection rule.
- `--frontend-strategy adaptive` tries descriptor-ranked candidates sequentially, exiting after the first **KISS + scan-GICP converged** hypothesis that also passes a provisional inlier count and fraction gate.
- `frontend_candidate_trials.csv` logs attempted candidate rank, reference, measured per-attempt wall time (including lazy loading), KISS and scan GICP reported times, inlier counts, and gate decision.
- `frontend_trials.csv` adds `frontend_strategy` and `frontend_candidates_attempted`.
- BEVPlace++ embeddings, map geometry, BST K=4, native-surfel optimizer, and causal database are unchanged.

**CAUTION:** The gate measures only inlier support, not normalized geometric residual or independent overlap verification. Inlier count is not a calibrated pose confidence and can accept an incorrect but structurally similar place. Therefore adaptive mode is experimental and must be judged by paired global, fine and strict 6-DoF success, not speed alone. No new algorithmic success claims until NCLT run.

**IMPORTANT:** `frontend_ms` and `total_ms` exclude online BEV image creation and network inference. `--frontend-pose-cache` is for local-solver replay and does not measure frontend latency. For a meaningful 9.7D experiment, do NOT use frontend pose replay.

## Install
Unzip from repository root and run:

```sh
python -m pytest -q tests/test_frontend_adaptive.py tests/test_native_surfel_registration.py tests/test_native_surfel_prepared.py tests/test_localization_refactor.py tests/test_orientation_diagnostics.py
```

## 1. Profile frozen exhaustive control (no change to candidate selection)

```sh
python run_localization.py \
  --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --local-solver native_surfel --frontend-strategy exhaustive \
  --output results/step9_7d_exhaustive
```

## 2. Test provisional top-1-first adaptive exit

```sh
python run_localization.py \
  --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --local-solver native_surfel --frontend-strategy adaptive \
  --output results/step9_7d_adaptive
```

The default gate (`--frontend-min-inliers 350`, `--frontend-min-inlier-fraction 0.025`) is a *starting diagnostic configuration*, not a validated/optimized threshold. Inference based on a 20,000-point downsample can change the meaning of a count. Don't adjust thresholds using evaluation ground truth and claim a held-out win; use a disjoint development split.

## Compare
- Confirm same 200 query identities, latest-prior reference catalogs and frozen map counts.
- Compare `frontend_candidate_trials.csv` to see candidate-rank runtime, failed KISS attempts and the distribution of how many attempts were avoided.
- Compare `frontend_trials.csv` for selected candidate rank, global 2m/5°, yaw and rotation; compare `localization_trials.csv` for paired fine/strict 6-DoF and regressions.
- Compare median/P95 frontend wall time, *and* per-stage KISS/GICP times. Add measured BEV preprocessing/inference and raw I/O for online timing.
- For acceptance require no statistically meaningful or practically unacceptable decrease in global/fine/strict success versus exhaustive, and substantial latency savings. Re-run identical configurations because historical experiments showed nondeterminism.

## What's next (separate step)
Implement and benchmark full BEVPlace++ local-feature pose estimation as a separate selectable frontend, with published checkpoint and output validation. No such pose model is silently substituted here, because current cached 8192-D descriptors alone do not contain the local feature grids required for matching.
