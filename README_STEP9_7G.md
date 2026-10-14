# Step 9.7G — BEV-first, adaptive-KISS fallback (experimental)

Patch on top of Step 9.7F **including the optional-backends hotfix**. This release adds `--frontend-strategy bev_hybrid`, a BEV local-feature confidence gate, and an adaptive-KISS fallback; the 6-DoF native surfel solver remains **unchanged**.

## Install

```sh
unzip -o Step9_7G_BEV_Hybrid.zip
python -m pytest -q tests/test_step97g.py tests/test_step97f.py tests/test_step97f_backend_wiring.py tests/test_frontend_adaptive.py tests/test_native_surfel_registration.py tests/test_native_surfel_prepared.py
```

## Run

```sh
python run_localization.py \
  --retriever bevplace \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --bev-local-feature-dir results/step9_7f_local_features \
  --cache-dir .cache/step8_1 \
  --top-k 5 \
  --frontend-strategy bev_hybrid \
  --hybrid-bev-min-inliers 28 \
  --hybrid-bev-min-ratio 0.07 \
  --hybrid-bev-max-candidates 2 \
  --local-solver surfel_6dof \
  --output results/step9_7g_bev_hybrid_6dof
```

Controls: `--frontend-strategy adaptive` and `--frontend-strategy bev_local` with `--local-solver surfel_6dof`, using same frozen queries, descriptors, surfels, and checkpoints. Do not supply `--frontend-pose-cache` to measure frontend runtime.

## Algorithm

1. Get frozen BEVPlace++ top-five candidates.
2. Try BEV local matching on **at most two** candidates in rank order. Confidence requires RANSAC convergence, >=28 inliers, and inliers / min(query feature count, reference feature count) >=0.07. Counts/ratio are provisional and *not* calibrated to pose correctness.
3. Accept the first BEV pose passing the gate; otherwise fall back to the original *adaptive KISS + scan-GICP* sequence, starting at rank 1. This can cost more than adaptive KISS on failures.
4. Run the unchanged native surfel 6-DoF optimizer for voxel and BST K=4.

`frontend_candidate_trials.csv` contains both BEV and KISS attempts with `backend` and `hybrid_bev_confident` columns (other columns are empty for whichever backend wasn't used). The `kiss_used` field shows fallback. Metadata records hybrid hyperparameters.

## Honest limitations

- **This is a first hybrid implementation, not a proven confidence estimator.** Incorrect BEV matches can be confidently accepted. Selection thresholds were not validated on held-out scans. Do not tune based on the same 200 test queries and then claim new generalization.
- The BEV extraction, model inference, and historical feature cache construction are **not included** in per-query latency. Time full online pipeline separately.
- BEV feature matching remains **scan-to-scan**. It is not yet feature matching against BST surfels.
- The initial roll/pitch/z for BEV hypotheses are inherited from the historical reference frame; strict 6-DoF accuracy depends on the refinement and geometry. No query ground truth should be used in initialization.
- The local 6-DoF optimizer (~180 ms measured in 9.7F) is deliberately unchanged. A separate performance-only optimization with fixed frontend pose replay would be required to isolate local solver speedups.
- The upstream coordinate convention test covers synthetic planar transform orientation; **empirical sign and feature-grid alignment on NCLT remain to be audited**. Compare per-query errors on a separate diagnostic subset, and verify against the official BEVPlace++ implementation.

## What to upload

For hybrid and adaptive-KISS controls: `frontend_candidate_trials.csv`, `frontend_trials.csv`, `localization_trials.csv`, `localization_summary.csv`, `representation_gates.csv`, `metadata.json`. Compare global 2m/5°, BST fine and strict 6DoF rates, candidate/fallback counts, frontend P50/P95, and total P50/P95. Check paired wins/losses.
