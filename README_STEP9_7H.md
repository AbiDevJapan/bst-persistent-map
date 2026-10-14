# Step 9.7H — Prepared full-6DoF surfels + phase profiling

Apply on top of **Step 9.7G** (`unzip -o Step9_7H_Prepared_6DoF.zip`). This patch modifies only `localization/surfel_6dof.py` and `benchmark/localization_benchmark.py` and adds tests.

## Behavior
- Builds surfel eigensystems once per historical voxel/BST map using `NativeSurfelConfig(min_horizontal_normal=0.0)`, preserving horizontal ground/roof normals needed for full 6DoF.
- Passes precomputed normals/validity into the original optimizer, leaving nearest-neighbor selection, damping, robust loss, step caps, acceptance, and pose fallback unchanged.
- Adds `surfel6d_normal_ms`, `surfel6d_tree_ms`, `surfel6d_correspondence_ms`, `surfel6d_linearize_ms`, `surfel6d_acceptance_ms` to `localization_trials.csv` for profiling.
- Creates a local KD-tree **per query**; that cost remains. Preparing normals adds RAM (~33 bytes per map surfel, plus overhead) and map-build time; account for them separately.
- Does not yet include compiled correspondence kernels. A correctness-preserving preparation optimization is intentionally isolated from numerical changes.

## Tests
`python -m pytest -q tests/test_step97h.py tests/test_step97f.py tests/test_step97g.py`

## Frozen-frontend benchmark (recommended)
Capture a control 9.7G `--frontend-strategy adaptive --local-solver surfel_6dof --save-frontend-poses` run **before applying the patch**, writing `results/step9_7h_pose_control`. Then replay the exact same frontend pose cache with the original solver in one worktree and 9.7H in another, both using `--frontend-pose-cache results/step9_7h_pose_control/frontend_poses.npz --local-solver surfel_6dof`. Preserve the same data, voxel/BST map and hardware. Replay totals exclude the global frontend.

If the prepatch baseline was not captured, use the Step9.7F control pose cache if it exists and was saved with `--save-frontend-poses`; otherwise establish a fresh frozen pose cache first.

## Criteria
- Compare each `(query_session, original_scan_index, map_geometry)` final pose error, convergence, score and inlier count against prepatch baseline (absolute differences near floating-point tolerance). Aggregate success counts alone are insufficient.
- Benchmark medians/P95 of `local_gicp_ms` for voxel and BST separately; report preparation time and memory.
- Inspect `surfel6d_*_ms` fields to decide whether to optimize local KD-tree / NN queries or linearization next.
- Current Step 9.7G BST baseline ~172 ms and strict-6D 57.5%. A 60–80 ms solver is **not guaranteed** by this preparation-only patch.
- Frontend retrieval, BEV/KISS candidate selection and original BST geometry remain untouched.
