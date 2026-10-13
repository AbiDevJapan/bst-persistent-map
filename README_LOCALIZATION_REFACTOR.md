# Lifelong LiDAR localization — consolidated Step 9.1–9.6 implementation

## What this changes

This is a **behavior-preserving refactor of the Step 9.6 global-to-local benchmark**, plus a Scan Context retrieval option. It is an overlay for the complete existing repository, **not** a replacement for `realdata/`, `representations/`, or `bst_id/`.

The new active pipeline is:

1. Load **exact frozen NCLT query scans** and causally latest-prior map session.
2. Build/load historical reference keyframes, preserving the Step 9.5 Scan Context cache location and ordering.
3. Retrieve top-five locations using the selected `BEVPlace++` or `scan_context` strategy. **No GT is used for retrieval**.
4. Use the **same** isolated KISS-Matcher, small_gicp scan refinement, and Scan Context yaw fallback when KISS fails.
5. Materialize the **same** frozen balanced geometry gate, **K=4** BST local map and 0.5 m voxel baseline.
6. Run both local-map GICP refinements from **the identical global 6-DoF hypothesis**.
7. Emit the same primary CSV files and Step 9.6-compatible result columns.

**Do not regenerate BEVPlace++ descriptors.** The pre-existing descriptor cache can be reused as is. This patch does not integrate the official BST-ID authors' codec; `PrototypeBSTSpatialCodec` remains unchanged.

## Install: overlay without deleting the original implementation

From the Git repository root (make a new branch/commit first):

```bash
unzip -o bst_localization_refactor.zip
python -m pytest -q tests/test_localization_refactor.py
python -c 'from benchmark.localization_benchmark import benchmark_localization; print("unified benchmark imports OK")'
```

**Important:** Do not delete old `step*.py` files yet. They remain useful for the original Git history, older tests, and baseline comparisons. The new production entry point does **not** import the old `localization.step*` or `benchmark.step*` modules. Historical source files may be removed in a later **separate** cleanup commit after the golden regression passes; other older runners/tests will need archiving or migration first.

## Execute BEVPlace++ (frozen Step 9.6 reproduction)

```bash
python run_localization.py \
  --retriever bevplace \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --descriptor-cache-dir results/step9_6_bevplace_descriptors \
  --cache-dir .cache/step8_1 \
  --top-k 5 \
  --output results/localization_refactored_bevplace
```

This uses the frozen BEVPlace++ checkpoint **through the cached descriptors**. There is no model loading or neural inference in this benchmark; like Step 9.6, time measurements **exclude** descriptor extraction and BEV preprocessing.

## Execute Scan Context (comparison)

```bash
python run_localization.py \
  --retriever scan_context \
  --data-root data \
  --query-scans-csv results/step9_5_global_to_local/query_scans.csv \
  --cache-dir .cache/step8_1 \
  --top-k 5 \
  --output results/localization_refactored_sc
```

This shares all downstream map and registration code with BEVPlace++. **SC timing differs from the old Step 9.5**: the consolidated runner counts failed KISS attempts rather than omitting them. Result quality and retrieved candidates should remain unchanged.

## Validate against the 200-query pre-refactor golden result

The archive includes the ten output files from the supplied Step 9.6 run and its original regression checker under `regression/golden_step96/`.

```bash
python check_localization_regression.py \
  --candidate results/localization_refactored_bevplace
```

**Must match exactly or within the checker's specified tolerances:** query identities, chronological history choices, reference database indices/counts, map geometry and primitive counts, selected historical reference IDs, and all 400 paired per-query success/failure flags. The checker **does not** enforce identical timing; runtime naturally varies. The reference model's source and descriptor provenance must also remain identical. If this check fails, do not present results as equivalent.

Note: This validation requires NCLT files and the original cached maps on your Mac. The complete 200-query NCLT registration benchmark could not be run in the packaging environment, which only has source archives and cached descriptor exports.

## New active files

| New file | Previous source / responsibility |
| --- | --- |
| `run_localization.py` | Single Step 9.5/9.6-compatible CLI, selectable retriever |
| `benchmark/localization_benchmark.py` | One shared causal map construction and 6-DoF registration loop from Step 9.6 |
| `benchmark/localization_inputs.py` | NCLT replay, scan preparation, and exact historical reference keyframe cache |
| `benchmark/localization_helpers.py` | Frozen K=4 profile, deterministic cap points, storage bytes, exclusions |
| `benchmark/localization_reports.py` | Pooled/per-session and retriever-specific metrics |
| `localization/retrieval.py` | SC/BEVPlace++ strategy selection, cache identity validation, no GT inputs |
| `localization/global_frontend.py` | Previously `step95.py`, including isolated KISS worker and small_gicp wrappers |
| `localization/bevplace.py` | Previously `step96_bevplace.py`, including pretrained descriptor cache validation |
| `localization/history.py` | Previously `step94.py`: causal historical map selection |
| `localization/pose.py`, `point_to_plane.py`, `pyramid.py`, `surfel_icp.py` | Moved historical 3-DoF geometry and scan preparation kernels, kept unchanged |
| `localization/surfels.py`, `geometry_gate.py`, `lod.py`, `compact_map.py` | Native surfel construction, balanced gate, K=4 hierarchy and compact map |
| `run_bevplace_extract.py`, `run_bevplace_retrieval.py` | Renamed optional descriptor extraction and retrieval-only diagnostics |
| `check_localization_regression.py` | Strict golden CSV/metadata comparison |
| `tests/test_localization_refactor.py` | Adapter, causal-profile, storage accounting, and legacy dependency isolation tests |

### Deliberately excluded from the refactor

- Changing the frozen balanced gate (sigma 0.05 m, relative planarity 0.025, normal coherence 0.95, max extent 2 m) or K=4.
- Changing deterministic point capping, reference sampling, GICP correspondence thresholds, candidate ordering, or yaw fallback.
- Training/fine-tuning BEVPlace++ or regenerating caches.
- Reimplementing the upstream mapping representations or replacing the prototype BST codec with author BST-ID.
- New precomputed map serialization: a later, independent optimization commit should introduce that with its own equivalence tests.

## Verification in this package

- Code import and compile tests pass.
- **123/123** previous project tests pass in the assembled source checkout.
- **7/7** added refactor tests pass.
- The full **200-scan NCLT replay has not been executed here**; run the golden checker locally before deleting legacy files or citing new scores.

## Git workflow

1. `git switch -c localization/unified-refactor`
2. Unzip the new source overlay and `git add` it. Commit this behavior-preserving migration only.
3. Run the full BEVPlace++ NCLT reproduction and golden check.
4. Only once it passes, remove/archive obsolete historical runners and files in a **second** commit.
5. Make any future map caching or timing optimizations in a **third** commit with separate benchmarks.
