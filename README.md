# BST-ID Lifelong Mapping — Step 2

Deterministic synthetic benchmark for testing the BST-ID lifelong-map idea.

This project intentionally contains **no BST-ID map representation yet**.
It creates the canonical ground truth that Step 3 will feed into:

1. snapshot-per-session storage,
2. Octree + temporal history,
3. BST-ID spatiotemporal storage.

## Default benchmark

- 100 m × 100 m × 5 m
- 0.5 m voxels
- 100 sessions
- coherent object-level changes
- fixed seed (`42`)
- static, slow, medium, and highly dynamic objects

Default per-session change probabilities:

| zone | probability |
|---|---:|
| static | 0.00 |
| slow | 0.03 |
| medium | 0.15 |
| high | 0.75 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

Generate 500 sessions:

```bash
python run_demo.py --sessions 500
```

Persist the canonical history:

```bash
python run_demo.py --sessions 100 --save generated/step2_100
```

## Tests

```bash
pytest -q
```

The key invariant is checked for every transition:

\[
M_t = (M_{t-1} - \Delta_t^-) \cup \Delta_t^+
\]

## Query API

```python
snapshot(dataset, t)

region_at_time(
    dataset,
    region=(xmin, xmax, ymin, ymax, zmin, zmax),
    t=t,
)

changes_between(dataset, region, t_a, t_b)

change_frequency_map(dataset)
```

`changes_between` is the **net state difference** between the two endpoint
sessions, matching the Step-1 experiment contract.

## Why object-level changes?

The benchmark moves or toggles cuboid objects rather than independently
flipping random voxels. A moving car therefore creates a coherent removal at
its old location and addition at its new location, which better resembles a
lifelong mapping problem.

## Step-2 completion criterion

The benchmark must answer exactly:

- what was occupied at session `t`,
- what was occupied inside a chosen region at `t`,
- what was added/removed between two sessions,
- which voxels changed most frequently,
- and reconstruct every session perfectly from its predecessor plus delta.

Once these hold, Step 3 can implement the three competing representations.


## Step 5

See `README_STEP5.md`. Run `python run_step5.py --sessions 10,25,50,100`.

## Step 6

See `README_STEP6.md` for strengthened Octree baselines and adversarial sweeps.

```bash
python run_step6.py --suite quick --sessions 100 --queries 4
```


## Step 7

See `README_STEP7.md` for the BST-specific isolation experiment.


## Step 8

See `README_STEP8.md` for NCLT real-data ingestion and benchmarking.

## Step 8.1

For million-voxel real NCLT runs, use `run_step8_1.py` and see `README_STEP8_1.md`. This replaces the object-heavy Step-8 benchmark path while retaining Step 8 for regression/reference.

## Step 8.2

See `README_STEP8_2.md` for the sparse-Octree baseline hardening experiment.

## Step 8.3

See `README_STEP8_3.md` for the N=2/3/5 real-session scaling experiment, DeltaVersion and packed voxel-history baselines, native-vs-decompressed query modes, and the isolated BST codec architecture.

See `BST_CODEC_INTEGRATION.md` for the future verified BST-ID replacement boundary.

## Step 8.4

See `README_STEP8_4.md` for the time-local adaptive tiling optimization. It uses the same five cached NCLT sessions as Step 8.3 and adds `time_local_explicit` plus `bst_time_local` without changing the isolated BST codec boundary.


## Step 8.5

See `README_STEP8_5.md` for the counted time-aware `(spatial bucket, session)` index over the exact Step-8.4 time-local block table. It keeps the union-index methods in the same run so index speed versus storage is isolated directly.
