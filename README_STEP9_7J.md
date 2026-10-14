# Step 9.7J — Parallel exact nearest-neighbor search for 6-DoF surfel registration

This is an opt-in speed experiment built on Step 9.7I. It preserves the original SE(3) point-to-plane objective, nearest-neighbor metric, correspondence radius, robust weighting and acceptance logic. The only changes are configurable SciPy cKDTree query worker count and leaf size. No approximate matching (`eps=0` remains the default). **Parallel search is not guaranteed to be faster**; use the profiling CSVs on the target Mac to select a configuration.

Install: extract ZIP at the root of a working Step 9.7I checkout (`unzip -o Step9_7J_Parallel_6DoF.zip`). It overwrites only `localization/surfel_6dof.py`, `benchmark/localization_benchmark.py`, `run_localization.py`, and adds `tests/test_step97j.py` and this README.

Test: `python -m pytest -q tests/test_step97j.py tests/test_step97i.py tests/test_step97h.py`

Benchmark with frozen frontend poses (the previous Step 9.7H cache) using the same common arguments as Step 9.7I, **with `--sc-top-k 5`**, `--local-solver surfel_6dof`, `--frontend-pose-cache results/step9_7h_prepared_6dof/frontend_poses.npz`, `--bst-xy-index` and `--surfel6d-fast-tree`:

```
python run_localization.py "${COMMON[@]}" --bst-xy-index --surfel6d-fast-tree --surfel6d-nn-workers 1 --surfel6d-leafsize 16 --output results/step9_7j_control
python run_localization.py "${COMMON[@]}" --bst-xy-index --surfel6d-fast-tree --surfel6d-nn-workers 2 --surfel6d-leafsize 16 --output results/step9_7j_workers2
python run_localization.py "${COMMON[@]}" --bst-xy-index --surfel6d-fast-tree --surfel6d-nn-workers 4 --surfel6d-leafsize 16 --output results/step9_7j_workers4
python run_localization.py "${COMMON[@]}" --bst-xy-index --surfel6d-fast-tree --surfel6d-nn-workers 2 --surfel6d-leafsize 32 --output results/step9_7j_workers2_leaf32
```

If COMMON is not defined in your shell, set it as a zsh array first; see Step9_7I README. Do not compare replay-only totals to live frontend runs. Run sequentially; the extra workers may suffer from oversubscription with BLAS/OpenMP. Keep the 1-worker mode if parallel overhead outweighs query speed. Confirm per-query pose, success, and inlier count equivalence, not just aggregate success. The indexed BST retrieval from Step 9.7I had three boundary mismatches and is not corrected by this patch.

Limitations: This patch does not reuse the local KD-tree across queries; local clouds are pose-centered and change. The dominant 68 ms correspondence cost may require a specialized compiled batch NN kernel for a larger reduction. Online BEVPlace++ inference is still excluded from reported benchmark times.
