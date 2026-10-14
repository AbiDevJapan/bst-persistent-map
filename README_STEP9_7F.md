# Step 9.7F — BEVPlace++ local-feature XY/yaw + 6-DoF native surfel optimization

## Deliverable

A separately selectable `--frontend-strategy bev_local` replaces KISS with FAST keypoints + pretrained BEVPlace++ local REIN descriptors + deterministic 2-point RANSAC; `--local-solver surfel_6dof` runs full six-axis native-surfel point-to-plane refinement on *both* voxel and BST maps. Adaptive KISS and 9.7C XY-yaw are retained unchanged.

**Status: code integration and synthetic tests only. NCLT data experiment and real network integration are not validated here.** Baselines (9.7D / 9.7E) cannot be substituted for measured 9.7F results.

**Important limitations**

* Local feature cache extraction runs separately, using the same trusted pretrained model checkpoint and voxelized 201×201 BEV raster from Step 9.6. Reference features are historical and reusable; query feature extraction is online cost, but *not counted* in the benchmark. Never describe the benchmark's `total_ms` as online latency.
* Matching uses the pretrained REIN local features sampled at FAST keypoints (as in upstream KITTI evaluation), not native surfels. No network fine-tuning, retraining or new learned representation.
* Relative pose is **SE(2)**. Elevation and roll/pitch are initialized by composing with the *reference keyframe's* historical pose, without query GT. This is imperfect especially when a query is far from the reference; robust 6-DoF surfel alignment may reject such initializations. We explicitly benchmark strict 6-DoF success.
* `bev_local` attempts candidates in BEVPlace++ descriptor ranking, accepts first passing a deterministic feature-inlier gate; if all fail, original Scan Context yaw fallback is used (not KISS). This fallback is not guaranteed to be accurate.
* Six-axis optimizer checks observability, bounds translation and rotation relative to initialization, and refuses invalid or unobservable refinements. If it fails, original frontend pose is kept. This is an accuracy-first Python baseline, not a tuned fast native/C++ GICP.
* `surfel_6dof` computes normals of local surfels per query, including ground/roof surfaces. 9.7C XY/yaw preparation cache's horizontal-normal gate is inappropriate for 6-DoF and therefore not reused. Memory cost for cached features must be reported separately.
* One particularly important gate: BEVPlace++ upstream REM feature output must be spatially aligned with the 201×201 raster. Extractor **raises** rather than silently interpolates when this differs.
* The code still uses PrototypeBSTSpatialCodec; it is not official BST-ID integration.

## Installation

Copy/extract this patch into the repository *already containing Step 9.7E and Step 9.6 model extraction files*. Do not replace the full repository with this patch.

```sh
python -m pytest -q tests/test_step97f.py \
  tests/test_frontend_adaptive.py tests/test_frontend_geometry_verifier.py \
  tests/test_native_surfel_registration.py tests/test_native_surfel_prepared.py
```

## Extract local features once

Supply the **same** checkpoint, BEVPlace++ checkout, Step9.6 descriptor cache, and NCLT data paths as used for the global descriptors:

```sh
python run_step9_7f_extract.py \
  --data-root data \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --bevplace-repo /path/to/BEVPlace2 \
  --checkpoint /path/to/model_best.pth.tar \
  --device mps \
  --max-keypoints 512 \
  --output results/step9_7f_local_features
```

Reference caches use descriptor row order; query caches use the same frozen query scan selection. Check extraction validation before running the full benchmark.

## Benchmarks

**Control:** Step9.7D adaptive KISS, using *the new same six-axis solver* to separate frontend from optimizer:

```sh
python run_localization.py --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 --top-k 5 \
  --frontend-strategy adaptive --local-solver surfel_6dof \
  --output results/step9_7f_kiss_6dof_control
```

**Experimental:** BEV local planar matching + same six-axis native solver:

```sh
python run_localization.py --retriever bevplace --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --bev-local-feature-dir results/step9_7f_local_features \
  --cache-dir .cache/step8_1 --top-k 5 \
  --frontend-strategy bev_local --local-solver surfel_6dof \
  --output results/step9_7f_bev_local_6dof
```

**Additional ablation:** Use `--frontend-strategy bev_local --local-solver native_surfel` to isolate the impact of adding six DoF to map refinement.

Compare exact query identity; global ≤2m/5°, fine XY≤0.5m yaw≤1°, strict SE(3) translation≤0.5m rotation≤2°, convergence, per-candidate inlier gate, unsuccessful local matches, P50/P95 timings. Measure *online* query BEV construction+neural inference as a separate stage; report cold and warm historical feature-cache memory. Use held-out chronological queries for final results; the prior 200 have been used repeatedly for development.

## Notes on performance

The step is primarily an **algorithmic evaluation**. The per-query matcher is Python and currently loops through up to 600 minimal RANSAC hypotheses per candidate. Further speedups require vectorization, compiled RANSAC or reuse of upstream matching, and prepared 6-DoF surfel geometry. New results may be worse than KISS; no performance improvement is claimed until actual NCLT tests.
