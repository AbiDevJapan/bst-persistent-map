# Step 9.7I — Warm-map BST XY indexing + optional faster local KD-tree build

Install on top of **Step 9.7H**, from your project root:

```bash
unzip -o Step9_7I_Indexed_BST_6DoF.zip
python -m pytest -q tests/test_step97i.py tests/test_step97h.py tests/test_step97g.py
```

This patch changes only `localization/compact_map.py`, `localization/surfel_6dof.py`, `benchmark/localization_benchmark.py`, `run_localization.py`, and adds this README and tests. It does **not** modify BST block geometry, BEV retrieval, KISS selection, 6-DoF objective or pose acceptance rules.

## Added command-line flags

- `--bst-xy-index`: create one `scipy.spatial.cKDTree` over the active historical session's materialized BST surfel centroid XY positions. Local circle queries use the exact original squared-distance predicate and sort row IDs ascending. The active historical snapshot is built according to the same causal latest-prior policy. An extra index is held in RAM; inspect `map_geometry_stats.csv` columns `bst_xy_index_extra_mib`, `bst_xy_index_prepare_ms`. The additional memory estimate is APPROXIMATE, not a direct RSS measurement.
- `--surfel6d-fast-tree`: use `cKDTree(..., balanced_tree=False, compact_nodes=False)` for each *local* 6-DoF registration search index. Correspondence query, `workers=1`, gating, objective, and iterations are unchanged. An unbalanced tree can query **slower**, so evaluate separately.

Neither flag changes the default 9.7H behavior.

## Scientific comparison (same pose cache)

Use the uploaded 9.7H `frontend_poses.npz` at `results/step9_7h_prepared_6dof/frontend_poses.npz`. Replace this path if necessary.

Run these three commands separately (not in parallel, so cache effects and CPU contention remain controlled):

```bash
COMMON='--retriever bevplace --data-root data --query-scans-csv results/step9_5_global_to_local/query_scans.csv --descriptor-cache-dir results/step9_6_bevplace_descriptors --cache-dir .cache/step8_1 --top-k 5 --frontend-strategy adaptive --local-solver surfel_6dof --frontend-pose-cache results/step9_7h_prepared_6dof/frontend_poses.npz'
python run_localization.py $COMMON --output results/step9_7i_control
python run_localization.py $COMMON --bst-xy-index --output results/step9_7i_bst_index
python run_localization.py $COMMON --bst-xy-index --surfel6d-fast-tree --output results/step9_7i_both
```

Run `--surfel6d-fast-tree` without `--bst-xy-index` as a fourth experiment if separating the factorial effects precisely.

## Validation

1. Join `localization_trials.csv` by `(query_session, original_scan_index, map_geometry)`.
2. Check per-query final XY, 3-D translation, yaw and rotation errors against control (tolerance 1e-8 for BST-only index; small floating numerical differences can occur from the alternative KD-tree tie order). Check success flags, convergence, and inlier count too.
3. Report median/P95 `local_map_retrieval_ms`, `local_gicp_ms`, `total_ms` for voxel and BST separately; verify the exact field names in your output CSV before plotting.
4. Report index construction time, extra memory estimate, plus warmed-map memory/RSS if available. Your BEV descriptor and prepared map RAM remain excluded from serialized-geometry accounting, and offline network inference remains excluded from replay timing.
5. This patch creates a **secondary Cartesian index** over BST geometry; if speedup is compelling, account for this auxiliary structure as part of the full BST system. Do not claim it's a free benefit of BST-ID encoding.

## Limits

This is a conservative first spatial-index optimization, **not** a compiled correspondence kernel. It can eliminate temporal bucket lookups and repeated ragged root expansion in the local query; its overhead may be large for 2 million surfels per historical session. The alternative local cKDTree constructor may or may not reduce total solver time; correspondence search still occurs on every iteration. Targets such as <10ms retrieval or <50ms native 6-DoF are NOT demonstrated until measured.

The upstream official BST-ID codec is still not integrated. All comparisons remain on the frozen prototype geometry and spatial codec.
