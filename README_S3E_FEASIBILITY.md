# BST-ID multi-robot map-transfer feasibility on S3E Playground 2

## Dataset and scope

**Recommended first real multi-robot dataset:** S3E (Feng et al., IEEE RA-L),
**S3Ev1/S3E_Playground_2** from `PengYu-Team/S3E` on Hugging Face.
- Three simultaneously recorded ground robots: Alpha, Bob, Carol.
- One ROS 2 SQLite bag `S3E_Playground_2.db3` with synchronized sensors.
- Three GT pose files (`alpha_gt.txt`, `bob_gt.txt`, `carol_gt.txt`).
- Official Hugging Face files page lists this directory at **6.73 GB**.
- Full S3E repository is approximately **278 GB**: do **not** download all of it.
- Three-robot results are **real recorded sensor/map content**, but the replay is
  offline and map poses are provided by GT; GPS is artificially perturbed only
  in the map request. **This is not a deployment, GPS-SLAM or scan registration benchmark**.

Official links:
- https://pengyu-team.github.io/S3E/
- https://huggingface.co/datasets/PengYu-Team/S3E/tree/main/S3Ev1/S3E_Playground_2
- https://huggingface.co/datasets/PengYu-Team/S3E

**Download footprint:** 6.73 GB for the bag + tiny GT/metadata. Keep roughly
30–50 GB free for downloads and processed stores; actual processed size varies
widely with scan stride, max range and map density. Approximately 16–32 GB of
RAM is a sensible starting point for these defaults, not a measured requirement.

## Installation and download

From the extracted repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install rosbags huggingface_hub

# Download only the metadata, three groundtruth files and three calibration files, to inspect first:
python download_s3e_playground2.py --output data/s3e --ground-truth-only

# To download the large 6.73 GB SQLite rosbag, plus the tiny metadata, GT and calibration files:
python download_s3e_playground2.py --output data/s3e
```

Alternatively, visit the Hugging Face files page above and download the five
files individually. No ROS 2 installation is required: `sqlite3` reads the
recording and `rosbags` decodes standard ROS 2 CDR PointCloud2 messages.

Your layout should be:

```
data/s3e/S3Ev1/S3E_Playground_2/
    S3E_Playground_2.db3
    metadata.yaml
    alpha_gt.txt
    bob_gt.txt
    carol_gt.txt
data/s3e/S3Ev1/Calibration/
    alpha.yaml
    bob.yaml
    carol.yaml
```

## 1. Inspect topics

```bash
python run_s3e_feasibility.py inspect \
 --db3 data/s3e/S3Ev1/S3E_Playground_2/S3E_Playground_2.db3
```

The importer detects **exactly one** `sensor_msgs/msg/PointCloud2` per robot.
If there are multiple, use a JSON file like:

```json
{"alpha":"/Alpha/velodyne_points","bob":"/Bob/velodyne_points","carol":"/Carol/velodyne_points"}
```

Pass `--topics-json topics.json` to `build`. The names here are *illustrative*;
copy actual names from `inspect`. Only standard CDR PointCloud2 is supported.

## 2. Calibrate the coordinate system before evaluating accuracy

Ground-truth lines must contain `time_sec, easting_m, northing_m, z_m,
qx, qy, qz, qw` (whitespace or commas). The importer linearly interpolates
position and spherical-interpolates orientation between samples. Observations
without a nearby reference pose are discarded. All three robots share a
common UTM-like metric frame; coordinates are shifted by one common origin.

For correct sensor-point placement, **supply one `T_pose_lidar` 4×4 rigid
transform per robot**, expressed in the reference pose frame, from the official
S3E calibration parameters (including the GT reference frame convention):

```json
{
  "alpha": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "bob": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "carol": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

**The identity values above are placeholder values, not verified S3E
calibration.** Replace them with correctly composed transforms before citing
alignment, map overlap or localization metrics. The official dataset has
calibration files in `S3Ev1/Calibration` (download those separately).

The `--assume-identity-extrinsics` switch exists **only** for a deliberately
unvalidated pipeline smoke test and is recorded in metadata. Do not report
its results as calibrated real-data map consistency.

## 3. Build 3 robot-owned chronological stores

```bash
python run_s3e_feasibility.py build \
 --db3 data/s3e/S3Ev1/S3E_Playground_2/S3E_Playground_2.db3 \
 --gt-dir data/s3e/S3Ev1/S3E_Playground_2 \
 --extrinsics-json s3e_extrinsics.json \
 --output results/s3e_stores \
 --voxel-size 0.75 --epochs 3 \
 --scan-stride 10 --max-scans-per-robot 280 --max-range 40 \
 --tile-side 16
```

For *format-only* debugging, use `--assume-identity-extrinsics` in place of
`--extrinsics-json`, but do not report resulting map alignment as valid.

It reads only LiDAR topics, transforms points using each robot's associated GT
pose and supplied extrinsics, builds **cumulative past-only map snapshots** for
each robot, and creates exact `VersionedStore` and Step-10 `Geometry4D K=4`
stores plus cached Morton spatial indexes. The bounded scan stride is an
**engineering downsampling choice**, not a physical-robot GPS simulation.

Outputs include `results/s3e_stores/ingestion.json` (frame, topics,
assumptions, per-robot accepted-scan counts and actual trajectory samples) and
`results/s3e_stores/stores/{alpha,bob,carol}/{voxel,geometry}`.

Note: the *voxel snapshots* are causal, but the compact 4D geometry is built
over the complete set of snapshots. For strict online evaluation, rebuild
geometry using **only versions available at each epoch**, and account for
construction cost. This initial benchmark is a storage/query feasibility test.

## 4. Run real-data ROI exchange

```bash
python run_s3e_feasibility.py benchmark \
 --store-root results/s3e_stores \
 --output results/s3e_exchange \
 --requests 90 --gps-sigma 3 --alignment-sigma 1 \
 --local-radius 20 --bandwidth-mbps 2
```

Request centers come from actual receiver-robot trajectories at recorded scan
times. A sender is chosen among peers based on proximity to **already
observed** sender trajectories, not by peeking into future maps. All methods
receive the same requests; ROI radius includes a conservative uncertainty
margin. A transferred ROI is compared exactly against the sender's original
version at that location.

Generated `trials.csv` has bytes, selection/serialization CPU, decoding CPU,
modeled link latency, nonempty target geometry, and ROI coverage; `summary.csv`
compares six matched baselines; `bandwidth_sweep.csv` evaluates modeled links.
`report.md` and `metadata.json` explain limitations and checks.

**Important:** Because three robots may not have geometrically overlapping
coverage on all trials, report the *nonempty target* fraction. Zero-content
responses count in total communication and cannot demonstrate useful map
sharing. Run with multiple region sizes and scene types, not only this sequence.

## 5. Decision thresholds (proposed, not established)

- Exact reconstruction: **100%** of ROI queries (non-negotiable correctness).
- K4 BST vs *content-addressed* voxel tiles: **at least 30% total wire savings**.
- BST vs Morton on identical geometry: test **spatial query time and memory**;
  byte-for-byte identical transfer is expected and **not** proof of BST novelty.
- No unacceptable CPU/latency inflation at 0.25–5 Mbps (measured locally;
  network portion remains a bandwidth-only model).
- **Not yet tested**: final real scan-to-map localization success (target >=95%
  of a paired voxel baseline), inter-robot map alignment without GT,
  actual radio transport, collaborative pose-graph consistency.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_s3e_feasibility.py \
 tests/test_collab_feasibility.py tests/test_step105b_geometry.py
```

The tiny generated `.db3`-shaped fixture exercises data ingestion, three
separate stores and map request protocols; it does not stand in for the real
6.73 GB S3E bag. On a machine with the downloaded dataset, run the commands
above and inspect `ingestion.json` for dropped scans and alignment assumptions.
