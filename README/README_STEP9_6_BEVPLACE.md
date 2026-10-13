# Step 9.6 — BEVPlace++ global retrieval replacement (frozen BST K=4)

## Purpose

Keep the Step 9.5 NCLT query set, causal latest-prior reference scan indices,
KISS-Matcher scan-to-scan registration, small_gicp refinement, frozen 0.5m voxel
baseline and balanced compact BST K=4 representation. Replace ONLY the initial
Scan Context candidate RANKING with official BEVPlace++ REIN-NetVLAD descriptors.
An additional Scan Context **yaw fallback** remains in place only when all KISS
candidates fail, to preserve Step 9.5's fallback philosophy. BEVPlace++ local
feature/RANSAC pose recovery is not introduced here; that would be a distinct
registration ablation and violate this initial controlled experiment.

## Upstream

Official BEVPlace++ implementation: https://github.com/zjuluolun/BEVPlace2
(T-RO 2025; pretrained checkpoint required; not bundled).
Upstream `REIN.py` returns `(..., local_feats, global_desc)`, where the global
NetVLAD descriptor is 8192-dimensional, and uses hardcoded `.cuda()` calls.
`datasets/gen_bev_images.py` defines the 0.4m-voxel 201x201 BEV density image.
The exporter reproduces that approach on the project's calibrated NCLT body
point clouds; compare generated BEVs on several scans with upstream before
interpreting research scores. The exact NCLT preprocessing is a research
adaptation and is not guaranteed pixel-identical to upstream NCLT benchmarks.

## Installation

Copy the contents of this archive into the **existing** project root. It is an
incremental patch, not a standalone replacement for the full Step 9.5 project.
Do not overwrite old Step 9.5 files. Model source/weights are not bundled.

In an appropriate separate Linux CUDA environment (PyTorch + torchvision, Open3D,
NumPy and your existing project's NCLT ingest modules), clone the official
BEVPlace2 repository and obtain its author-trained `model_best.pth.tar` or
`checkpoint.pth.tar`. The checkpoint MUST be for BEVPlace++, not older BEVPlace.
Verify the checkpoint source yourself; the extractor uses `weights_only=True`
where supported and loads the state dict with `strict=True`. It refuses to
continue if CUDA is absent (official code assumes CUDA).

```bash
git clone https://github.com/zjuluolun/BEVPlace2.git third_party/BEVPlace2
# install a compatible CUDA torch/torchvision and open3d in your env
pytest -q tests/test_step96_bevplace.py
```

If NCLT, checkpoint, and the Step 9.5 reference caches are located on different
machines, run descriptor extraction where all are accessible and copy only the
resulting `reference_*.npz` and `query_*.npz` to the benchmark machine.

## 1. Export the same 200 query scans and prior keyframe descriptors

Use the exact original Step 9.5 `query_scans.csv` and reference cache directory.
It uses 50 scans from each of Feb/Mar/May/June, and Jan/Feb/Mar/May causal
latest-prior reference keyframes, respectively.

```bash
python run_step9_6_extract.py \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --reference-cache-dir .cache/step8_1/step9_5_scan_context \
  --bevplace-repo third_party/BEVPlace2 \
  --checkpoint /absolute/path/to/model_best.pth.tar \
  --batch-size 4 \
  --output results/step9_6_bevplace_descriptors
```

Files are tied to the checkpoint SHA256, preprocessing ID, and exact original
scan-index mapping. We refuse mismatches rather than silently retrieving against
wrong poses/keyframes. The query encoder does not access/query GT poses.

## 2. Retrieval-only comparison (recommended first)

Run this on CPU; it reads NCLT GT **only to score** retrieval correctness:

```bash
python run_step9_6_retrieval.py \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --output results/step9_6_bevplace_retrieval
```

Reports `retrieval_trials.csv`, `retrieval_summary.csv` and `metadata.json`,
including oracle database coverage@5/10/20m and Recall@1/5/10/20 at each distance.
No registration or BST maps are constructed for this phase.

Compare BEVPlace++ Recall@5 with frozen Scan Context 48.5% on the same 200 queries,
not the official paper's NCLT recall (different evaluation protocol).

## 3. Full paired voxel-vs-BST validation

```bash
python run_step9_6.py \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --sc-top-k 5 \
  --output results/step9_6_bevplace_global_to_local
```

`--sc-top-k 5` is a retained Step 9.5 compatibility flag, now controlling the
BEVPlace++ top-K ranking. Start with K=5 for a paired method-only comparison.
Run K=20 as a separate compute/recall ablation and report runtime increased by
extra KISS calls. `--sc-candidate-pool` is ignored by BEVPlace++ ranking.

## Important limitations / claims

1. Descriptor extraction is **offline**. `frontend_ms` in the Step 9.6 paired
   run includes cached descriptor retrieval, all attempted KISS registrations
   (including failed attempts), lazy reference scan loading, and scan GICP, but
   **excludes BEV preprocessing, GPU network inference and cache building**.
   Consequently the exported `total_ms` is *not* genuine online latency. A later
   on-device end-to-end timing experiment must account for these costs.
2. Scan Context reference descriptors are still loaded/built for KISS-failure
   yaw fallback, not for the learned global candidate ranking.
3. Model weights must be pretrained on external data (e.g., author-provided KITTI
   weights), not on these 200 NCLT evaluation queries. Checkpoint identity is
   recorded in every cache file.
4. This is **not** the upstream paper's global pose estimator; it uses their
   learned place descriptor followed by the original frozen KISS + GICP stack.
5. The `PrototypeBSTSpatialCodec` remains in use: this does not complete the
   official authors' separate BST-ID codec integration.
6. The 200 paired scans are for continuity. If model selection/tuning uses them,
   obtain a new held-out query set for final independent paper validation.
7. The script does not benchmark successful online BEV extraction on macOS:
   unmodified author REIN runs CUDA; use a CUDA machine to export the descriptors
   and run the registration benchmark wherever KISS/small_gicp are installed.

## What success looks like

First: improved Recall@5m, identical prior reference coverage and filenames,
and zero future-session leakage. Second: higher KISS+GICP global 2m/5deg success
on paired scans. Third: BST representation gate maintained (>=90% fine success
retention, >=40% primitive reduction) and report strict 6-DoF separately.
