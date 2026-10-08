# BST Lifelong Mapping — Step 9.1.3

This package contains the cumulative experimental code through **Step 9.1.3**.

Step 9.1.3 freezes the Step-9.1.2 **balanced K=4** localization LOD and converts it into a **standalone materialized runtime table**. The binary merge hierarchy is now construction-only: after materialization/equivalence verification it is not required by the deployed localization path.

This step also corrects localization-memory accounting by counting the exact sufficient statistics consumed by the unchanged coarse-to-fine matcher (`count + centroid + covariance`) and the complete root directory, including global BST block IDs.

Start with:

- `README_STEP9_1_3.md` — current experiment, memory accounting, equivalence checks, and commands;
- `run_step9_1_3.py` — current entry point;
- `README_STEP9_1_2.md` — hierarchical K sweep that selected K=4;
- `README_STEP9_1_1.md` — geometry-aware two-level fallback experiment;
- `README_STEP9_1.md` — native block-surface feasibility baseline;
- `README_STEP9_0_3B.md` — validated voxel-surfel geometry baseline;
- `BST_CODEC_INTEGRATION.md` — isolated BST codec boundary.

No new NCLT sessions are required. Existing `.cache/step8_1` occupancy data is reused. Surfel statistics and the temporary merge hierarchy are rebuilt in memory and are not persisted by default.

For implementation equivalence, replay the Step-9.1.2 scans. For unbiased validation, use `--exclude-query-scans-csv` and request a larger held-out set; do not retune the balanced gate or K=4.
