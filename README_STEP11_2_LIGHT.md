# Step 11.2 (light) — Correction locality sanity check

**Purpose.** Confirm correction correctness and inspect how the same **eight** pose corrections behave as more *unchanged* NCLT scans enlarge the map. This is a quick go/no-go diagnostic, **not a paper-level benchmark**. No full-disk rebuild or multi-method tuning.

## Installation

Extract this patch at the root of an existing Step 11.1 project (including the ground-truth NaN fix). It adds only:

- `run_correction_scaling.py`
- `tests/test_step112_light.py`
- `README_STEP11_2_LIGHT.md`

Run `python -m pytest -q tests/test_step112_light.py tests/test_step111_nclt.py`.

## Quick smoke test using your existing 64-scan preparation

This reuses `results/step11_1_prepared` and runs only 3 small prefixes. Because the previously selected corrected scans are indices 16–23, the first prefix must include at least 24 scans.

```bash
python run_correction_scaling.py \
  --prepared results/step11_1_prepared \
  --output results/step11_2_quick \
  --sizes 24 40 64 --repeats 1
```

## Recommended larger NCLT check (prepare once)

Use the original session and unchanged perturbation window; merely raise the total selected scans to 256.

```bash
python run_nclt_correction.py prepare \
  --data-root data \
  --sessions 2012-03-17 \
  --output results/step11_2_prepared \
  --scan-stride 20 \
  --max-scans-per-session 256 \
  --corrupt-from 16 \
  --corrupt-count 8 \
  --translation 1.0 0 0 \
  --yaw-deg 2.0

python run_correction_scaling.py \
  --prepared results/step11_2_prepared \
  --output results/step11_2_scaling \
  --sizes 64 128 256 --repeats 1
```

**Outputs:** `trials.csv` and `summary.json`. Check `all_exact`, `all_previous_revisions_preserved`, affected/active tile counts, and incremental commit time as the map grows. A correction cost that remains fairly stable while map size increases supports update locality. If it grows with map size, inspect SQLite counters and per-source dependencies next.

**Timing boundaries:** `incremental_commit_ms` times committed SQLite correction only (no clone, no rebuild, no final query); `full_rebuild_ms` times independent in-memory voxel-union reconstruction only. Their absolute ratio is NOT a fair persistent-update speedup. The rebuild is used primarily as an exactness oracle and a rough scaling reference.

**Why fixed-size prefixes?** The script enforces that all corrected source IDs are present in every prefix and identical across configurations; it does not perturb additional scans as the map grows. Larger maps may still increase the number of *other robots'/scans' overlaps* in the corrected region, which is natural for this workload.

**Next milestone:** If correctness and update locality look good, stop benchmarking and work on the persistent spatial index / compressed provenance and history. Save large-scale, repeated comparisons against state-of-the-art systems for later.
