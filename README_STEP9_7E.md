# Step 9.7E — Independent Geometry Verification for Adaptive KISS

## Scope
Extends 9.7D with `--frontend-strategy adaptive_geo`. The previous `exhaustive` and `adaptive` modes are preserved. This is a *stronger early-exit gate*, not a trained confidence estimator, not an implementation of BEVPlace++ pose matching, and **does not guarantee an accuracy improvement**.

The new gate is evaluated only after KISS and scan-to-scan GICP converge and meet the original inlier thresholds. It independently transforms a bounded, deterministic sample of query points using the candidate's scan-GICP SE(3) transform, then matches each to the reference point cloud by nearest-neighbor search. It accepts an early exit only when BOTH:

- Independent NN overlap at 1.0 m is at least 55%.
- The best 80% of overlapping NN residuals have RMSE <= 0.65 m.

Thresholds are provisional engineering defaults. They are not calibrated on held-out NCLT sessions. These checks use *no query ground truth*. Wrong-place structural aliasing remains possible.

**Read this carefully**: Even a strict gate can worsen localization by rejecting a good candidate and selecting a worse later one (or by increasing latency). `adaptive_geo` is an **experiment**, not a recommended production default.

## Install
From repository root on the Step 9.7D+9.7C version:

```
unzip -o Step9_7E_Geometric_Verification.zip
python -m pytest -q tests/test_frontend_geometry_verifier.py tests/test_frontend_adaptive.py tests/test_native_surfel_registration.py tests/test_native_surfel_prepared.py tests/test_localization_refactor.py tests/test_orientation_diagnostics.py
```

## Comparison
Use the **same frozen 200 query scans**, descriptor cache, local solver and hardware setup. Do not provide `--frontend-pose-cache`, as that bypasses the work being benchmarked.

### Step 9.7D old adaptive control

```
python run_localization.py \
  --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --local-solver native_surfel --frontend-strategy adaptive \
  --output results/step9_7e_adaptive_control
```

### Step 9.7E independent geometry verifier

```
python run_localization.py \
  --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --local-solver native_surfel --frontend-strategy adaptive_geo \
  --output results/step9_7e_adaptive_geo
```

Optional: `--frontend-geo-radius-m 1.0`, `--frontend-geo-min-overlap 0.55`, `--frontend-geo-max-rmse-m 0.65`. Keep these fixed during the first evaluation. Do NOT tune against the same 200 scans and then claim held-out accuracy.

## Outputs
`frontend_candidate_trials.csv` has the original candidate timings and new fields:

- `inlier_gate_pass`: original 9.7D convergence/inlier test
- `geometry_gate_pass`: independent geometry thresholds passed
- `geometry_overlap`: fraction of independent query points within NN radius
- `geometry_trimmed_rmse_m`: trimmed residual among radius-overlapping samples
- `verified_for_early_exit`: the actual gate used for this strategy

Geometry fields are NaN when the geometric test is not executed. The candidate list/old fallback logic, chosen-pose scoring among attempted candidates, downstream native BST and voxel maps, and frozen K=4 map are unchanged.

## Evaluate and do not cherry-pick
Compare candidate counts, early stop rank, selected references, global 2m/5deg, BST fine XY<=0.5m and yaw<=1deg, strict 6DoF, median/P95 frontend and BST total latency, and the paired discordant success cases. Preprocessing and online BEV neural inference are not included in reported timings. Record extra validation milliseconds explicitly through per-candidate wall time; do not assume stronger checking is free.

**Scientific guardrail:** `adaptive_geo` may not recover any of the old adaptive-vs-exhaustive lost queries. Its first purpose is to measure whether an independent geometry gate can reject harmful wrong early exits at acceptable latency cost. Next step, if needed, is a calibrated/learned verifier trained on separate sessions or BEVPlace++ local-feature matching.
