# Lifelong NCLT Map Versioning and Selective Synchronization (Step 10)

**Primary workflow:** `README_STEP10.md`; `run_versioned_map.py`, `versioned_map/`.

**Optional legacy localization:** `run_localization.py`, `localization/`, `benchmark/`. See `README_STEP9_7J.md`.

**BST-ID codec selection:** `--codec official | prototype | morton`. Official author implementation is vendored under `third_party/bstid_author/bst_id`; legacy 3D prototype remains available unchanged. Both are separated from exact metric voxel contents.
