from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path

from realdata.nclt import NCLTIngestConfig
from realdata.nclt_packed import prepare_nclt_packed
from benchmark.step83_benchmark import benchmark_packed_step83

DEFAULT_PILOT='2012-01-15,2012-02-02,2012-04-29,2012-05-26,2012-08-04'

def parse_roi(s):
    if not s:return None
    v=tuple(float(x) for x in s.split(','))
    if len(v)!=6:raise ValueError('--roi needs xmin,xmax,ymin,ymax,zmin,zmax')
    return v

def parse_ints(s): return tuple(int(x.strip()) for x in s.split(',') if x.strip())

def main():
    p=ArgumentParser(description='Step 8.3: NCLT scaling + delta/voxel-history baselines + isolated BST codec')
    p.add_argument('--data-root',required=True)
    p.add_argument('--sessions',default=DEFAULT_PILOT)
    p.add_argument('--scales',default='2,3,5',help='Nested session-prefix scales, e.g. 2,3,5 or 2,3,5,10')
    p.add_argument('--voxel-size',type=float,default=.5)
    p.add_argument('--scan-stride',type=int,default=10)
    p.add_argument('--min-range',type=float,default=2)
    p.add_argument('--max-range',type=float,default=80)
    p.add_argument('--min-points-per-voxel',type=int,default=1)
    p.add_argument('--max-pose-gap-us',type=int,default=100000)
    p.add_argument('--max-scans-per-session',type=int,default=None)
    p.add_argument('--roi',default=None)
    p.add_argument('--cache-dir',default='.cache/step8_1')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--flush-scans',type=int,default=100)
    p.add_argument('--progress-every',type=int,default=250)
    p.add_argument('--queries',type=int,default=20)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--query-seed',type=int,default=42083)
    p.add_argument('--targeted-fraction',type=float,default=.75)
    p.add_argument('--include-full',action='store_true')
    p.add_argument('--output-modes',choices=['both','voxels','native'],default='both')
    p.add_argument('--preprocess-only',action='store_true')
    p.add_argument('--output',default='results/step8_3_nclt_scaling')
    a=p.parse_args()

    names=[x.strip() for x in a.sessions.split(',') if x.strip()]
    scales=parse_ints(a.scales)
    if names != sorted(names): print('WARNING: sessions are not chronological; temporal history follows supplied order.',flush=True)
    if max(scales)>len(names): raise ValueError(f'largest --scales value {max(scales)} exceeds supplied session count {len(names)}')

    cfg=NCLTIngestConfig(
        voxel_size_m=a.voxel_size,scan_stride=a.scan_stride,min_range_m=a.min_range,max_range_m=a.max_range,
        min_points_per_voxel=a.min_points_per_voxel,max_pose_gap_us=a.max_pose_gap_us,
        max_scans_per_session=a.max_scans_per_session,roi_local_m=parse_roi(a.roi),
    )
    print('Step 8.3 sessions:',', '.join(names),flush=True)
    print('Scaling prefixes:',scales,flush=True)
    prepared=prepare_nclt_packed(a.data_root,names,cfg,None if a.no_cache else a.cache_dir,a.flush_scans,a.progress_every)
    print(f'Prepared packed grid: {prepared.spec.nx} x {prepared.spec.ny} x {prepared.spec.nz}; origin={prepared.origin_local_m}',flush=True)
    for st in prepared.session_stats:
        print(f'  {st.session_name}: scans {st.scans_used}/{st.scans_selected}; occupied={st.occupied_world_voxels:,}; pose_gap_p95={st.pose_gap_us_p95/1000:.1f} ms',flush=True)
    if a.preprocess_only:
        print('Preprocess/cache complete; benchmark skipped.',flush=True);return

    modes=('voxels','native') if a.output_modes=='both' else (a.output_modes,)
    benchmark_packed_step83(prepared,Path(a.output),scales,a.queries,a.warmups,a.repetitions,a.query_seed,a.targeted_fraction,a.include_full,modes,True)
    print(f'\nWrote Step 8.3 results to {a.output}',flush=True)

if __name__=='__main__':main()
