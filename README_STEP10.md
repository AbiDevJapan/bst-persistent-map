# Step 10 — Versioned map + selectable BST-ID labels

This is a **non-destructive architectural refactor** of the full Step 9.7J source distribution: prior runnable localization code remains in `localization/`, `benchmark/` and `run_localization.py`, while the new *independent* primary workflow lives in `versioned_map/` and `run_versioned_map.py`. No neural model or KISS dependency is required to run Step 10.

Author source copied, unmodified, to `third_party/bstid_author/bst_id` from user-supplied `bst-id-ijgi-2026-major-revision.zip` (licensed use by user assertion). This avoids a namespace collision with the project's `bst_id` module. Use `--codec official` (default), `--codec prototype` to fall back, or `--codec morton` as a control.

## Install & validate

```sh
python -m pip install numpy pytest
python -m pytest -q tests/test_step10_versioning.py
```

## Ingest five NCLT sessions

```sh
python run_versioned_map.py build --data-root data --cache-dir .cache/step8_1 --database results/step10_official --codec official --tile-side 32
python run_versioned_map.py build --data-root data --cache-dir .cache/step8_1 --database results/step10_prototype --codec prototype --tile-side 32
python run_versioned_map.py build --data-root data --cache-dir .cache/step8_1 --database results/step10_morton --codec morton --tile-side 32
```

**Important**: identical ingest input/config MUST be used across runs, and build a new directory per codec. Full NCLT rebuild may take time and disk. For development, use `--max-scans-per-session 50` then remove it for the final evaluation. `--sessions` can override the default five dates.

## Historical queries in integer voxel coordinates

Bounds are six **half-open** integers `x0 x1 y0 y1 z0 z1`, relative to the canonical global voxel grid; physical positions are `origin_local_m + voxel_indices*voxel_size`.

```sh
python run_versioned_map.py stats --database results/step10_official
python run_versioned_map.py query --database results/step10_official --session 2012-03-17 --bounds 100 200 100 200 0 30 --save-npy results/march_roi.npy
python run_versioned_map.py changes --database results/step10_official --from-session 2012-02-02 --to-session 2012-05-11 --bounds 100 200 100 200 0 30
python run_versioned_map.py sync --database results/step10_official --from-session 2012-02-02 --to-session 2012-05-11 --bounds 100 200 100 200 0 30 --output results/step10_sync_packet.json
```

`sync` writes a transport JSON demonstration packet, not an optimized binary wire protocol. `apply_packet()` supports receiver-side reconstruction from the exact source version. For realistic bandwidth measurements, implement binary packaging with per-blob checksums and compare end-to-end sender/receiver protocol costs. Partial tile updates are intentionally transferred as whole tile blobs; outside-ROI changes may be included, and this should be reported.

## Optional localization

Existing `run_localization.py` remains untouched. It uses the **old localization-specific BST/voxel geometry store** until a separately validated adapter is developed. `--codec official` currently controls Step 10 tile labels only. Do not claim the Step 9 localization compact-map representation already uses the author's codec. Run your existing Step 9.7J command for localization. The new versioned database has no dependency on it.

## What was deliberately NOT implemented

- Replacing the old `PrototypeBSTSpatialCodec` inside adaptive 3D geometry with geospatial author IDs without a proven scale/domain transform.
- An upstream-prefix spatial query accelerator; the upstream codec's Web Mercator footprint needs separate tests.
- Independent-snapshot and Morton optimized binary baselines, operational storage ablations, full packet receiver CLI, or NCLT end-to-end experiment runs.
- Physics-based change detection; these maps only record observed occupied endpoints.

These are follow-up tasks, not measured results. `STEP10_EXPERIMENT_SPEC.md` freezes what must be tested.
