"""Multi-peer communication feasibility replay over *existing* Step 10 stores.

This is a map-exchange experiment with a coarse GPS prior. It does NOT run
simultaneous physical robots, an online distributed SLAM backend, map alignment,
or downstream scan registration. Exact endpoint equivalence is verified.
"""
from __future__ import annotations
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .protocol import METHODS,Receiver,Sender
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D


def _bounds(center,radius,dims,voxel_size):
    p=np.asarray(center)/voxel_size
    q=int(math.ceil(radius/voxel_size))
    low=np.floor(p).astype(int)-q
    high=np.floor(p).astype(int)+q+1
    return tuple(int(v) for j in range(3) for v in (low[j],high[j]))


def _source_point(store,session,rng):
    keys=list(store.all_versions()[session]);side=int(store.meta()['tile_side'])
    if not keys:raise ValueError('empty source map')
    tile=keys[int(rng.integers(len(keys)))];coords=np.asarray([int(x) for x in tile.split(',')])
    voxel=coords*side+int(side//2)
    return voxel.astype(float)*float(store.meta()['voxel_size_m'])


def create_tasks(sources, count, gps_sigma_m,local_radius_m,alignment_sigma_m,seed):
    rng=np.random.default_rng(seed);n=len(sources)
    if n<1:raise ValueError('no map sources')
    sessions=[list(db.all_versions()) for db,_ in sources]
    n_epochs=min(len(s) for s in sessions)
    if count<n_epochs:raise ValueError('requests must be at least number of epochs')
    saved={};tasks=[]
    # First sample by chronological epoch, without using future map content.
    for i in range(count):
        epoch=min(n_epochs-1, i*n_epochs//count)
        source=i%n
        receiver=(source+1+(i//n)%(max(1,n-1)))%max(2,n)
        if receiver==source:receiver=(source+1)%max(2,n)
        db,_=sources[source];version=sessions[source][epoch]
        key=(source,receiver)
        repeat=key in saved and i%5==4
        if repeat:
            ctr,true_ctr=saved[key]
        else:
            true_ctr=_source_point(db,version,rng)
            ctr=true_ctr.copy()
            ctr[:2]+=rng.normal(0,gps_sigma_m,2)
            ctr[2]+=rng.normal(0,alignment_sigma_m)
            saved[key]=(ctr,true_ctr)
        uncertainty=3.0*math.sqrt(gps_sigma_m**2+alignment_sigma_m**2)
        dims=db.meta()['dims'];vs=db.meta()['voxel_size_m']
        requested=_bounds(ctr,local_radius_m+uncertainty,dims,vs)
        target=_bounds(true_ctr,local_radius_m,dims,vs)
        # Guaranteed geometric containment of requested bounding volume is
        # stronger than a test that accidentally has no occupied voxels.
        covered=all(requested[j]<=target[j] and requested[j+1]>=target[j+1] for j in (0,2,4))
        tasks.append({'request_id':i,'source_robot':source,'receiver_robot':receiver,
                      'epoch':epoch,'version':version,'bounds':requested,'target_bounds':target,
                      'gps_offset_m':float(np.linalg.norm(ctr[:2]-true_ctr[:2])),
                      'gps_roi_encloses_target':covered,'reused_center':repeat})
    return tasks


def _summarize(rows):
    result=[]
    for method in METHODS:
        r=[x for x in rows if x['method']==method]
        def med(k):return float(np.median([x[k] for x in r]))
        def mean(k):return float(np.mean([x[k] for x in r]))
        result.append({'method':method,'requests':len(r),
                       'total_wire_bytes':sum(x['wire_bytes'] for x in r),
                       'mean_wire_bytes':mean('wire_bytes'),
                       'median_wire_bytes':med('wire_bytes'),
                       'p95_wire_bytes':float(np.percentile([x['wire_bytes'] for x in r],95)),
                       'payload_bytes_total':sum(x['body_bytes'] for x in r),
                       'headers_bytes_total':sum(x['header_bytes'] for x in r),
                       'median_selection_encoding_ms':med('selection_encoding_ms'),
                       'median_receiver_decoding_ms':med('receiver_decoding_ms'),
                       'median_end_to_end_local_ms':med('local_total_ms'),
                       'median_modeled_total_ms':med('modeled_total_ms'),
                       'p95_end_to_end_local_ms':float(np.percentile([x['local_total_ms'] for x in r],95)),
                       'mean_primitives':mean('primitives'),
                       'exact_success_rate':mean('correct_reconstruction'),
                       'gps_coverage_rate':mean('gps_roi_encloses_target'),
                       'receiver_cache_hit_rate':mean('cache_hit'),
                       'all_correct':all(x['correct_reconstruction'] for x in r)})
    ref=next(x for x in result if x['method']=='voxel_tiles')
    for x in result:
        x['wire_ratio_vs_voxel_tiles']=x['total_wire_bytes']/max(1,ref['total_wire_bytes'])
        x['wire_saving_vs_voxel_tiles']=1-x['wire_ratio_vs_voxel_tiles']
    return result


def benchmark(sources,output,*,requests=36,gps_sigma_m=3.0,
              alignment_sigma_m=1.0,local_radius_m=6.0,seed=20261015,
              bandwidth_mbps=2.0,verify=True,tasks=None,dataset_meta=None):
    if gps_sigma_m<0 or alignment_sigma_m<0 or local_radius_m<=0:raise ValueError('invalid geometry')
    if bandwidth_mbps<=0:raise ValueError('bandwidth must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    built_indexes=[]
    for j,(src,geom) in enumerate(sources):
        sessions=list(src.all_versions())
        if sessions!=geom.meta['sessions']:
            raise ValueError('source versions and geometry history differ; build complete geometry history')
        if src.meta()['dims']!=geom.meta['source_meta']['dims']:
            raise ValueError('mismatched coordinate bounds')
        if geom.meta['k']!=4:raise ValueError('this benchmark requires exact K=4 geometry')
        for session in sessions:
            index=src.root/'indexes'/f'{session}_morton_hash.npz'
            if not index.exists():src.build_spatial_index(session,'morton_hash');built_indexes.append(f'{j}:{session}')
        # Force index load before timed requests.
        geom._load_index('bst');geom._load_index('morton')
    tasks=(create_tasks(sources,requests,gps_sigma_m,local_radius_m,alignment_sigma_m,seed)
           if tasks is None else tasks)
    if len(tasks)!=requests:raise ValueError("number of explicit tasks differs from requests")
    receivers=defaultdict(Receiver);senders=[Sender(*s) for s in sources]
    rows=[];losses=[]
    nonempty_targets=0;nonempty_requests=0
    for task in tasks:
        i=task['request_id'];s=task['source_robot'];d=task['receiver_robot']
        src,_=sources[s];version=task['version'];bounds=task['bounds']
        # Untimed gold reconstruction is NOT added to method runtimes.
        expected=src.query(version,bounds,index_mode='morton_hash')
        target=src.query(version,task['target_bounds'],index_mode='morton_hash')
        nonempty_targets+=int(len(target)>0)
        nonempty_requests+=int(len(expected)>0)
        missing=np.setdiff1d(target,expected,assume_unique=True)
        task_coverage=bool(not len(missing))
        if task['gps_roi_encloses_target'] and not task_coverage:raise AssertionError('ROI containment invariant')
        methods=METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]
        for method in methods:
            recipient=receivers[(s,d,method)]
            wire,ss=senders[s].send(method,version,bounds,recipient)
            reconstructed,decode_ms=recipient.apply(wire,src._spec(),expected_bounds=bounds)
            eq=np.array_equal(reconstructed,expected)
            if verify and not eq:raise AssertionError(f'wrong receiver map: {method} {i} {version}')
            # Simulated serialization + raw bandwidth; excludes real transport,
            # contention, ROS2 overhead, CPU queues and real localization.
            transfer_ms=8.0*len(wire)/(bandwidth_mbps*1_000_000)*1000
            rows.append({'request_id':i,'method':method,'source_robot':s,'receiver_robot':d,
                         'version':version,'epoch':task['epoch'],
                         'bounds':','.join(map(str,bounds)),
                         'wire_bytes':len(wire),'body_bytes':ss['body_bytes'],
                         'header_bytes':ss['header_bytes'],
                         'selection_encoding_ms':ss['request_select_encode_ms'],
                         'receiver_decoding_ms':decode_ms,
                         'local_total_ms':ss['request_select_encode_ms']+decode_ms,
                         'modeled_link_ms':transfer_ms,
                         'modeled_total_ms':ss['request_select_encode_ms']+transfer_ms+decode_ms,
                         'primitives':ss['primitives'],
                         'cache_hit':int(ss['body_bytes']==0),
                         'index_ms':ss.get('index_ms',0.),'index_probes':ss.get('index_probes',0),
                         'correct_reconstruction':int(eq),
                         'roi_voxels':len(expected),'target_voxels':len(target),
                         'gps_offset_m':task['gps_offset_m'],
                         'gps_roi_encloses_target':int(task['gps_roi_encloses_target']),
                         'source_target_voxel_coverage':int(task_coverage),
                         'reused_center':int(task['reused_center'])})
        losses.append({'request_id':i,'gps_roi_encloses_target':task['gps_roi_encloses_target'],
                       'source_target_voxel_coverage':task_coverage,'target_voxels':len(target),
                       'gps_offset_m':task['gps_offset_m']})
    summary=_summarize(rows)
    with (output/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # Analyze which throughput makes the extra CPU overhead worthwhile,
    # without claiming any network performance was actually measured.
    sweep=[]
    for mbps in (0.25,0.5,1.0,2.0,5.0,10.0,25.0):
        for mode in METHODS:
            matches=[x for x in rows if x['method']==mode]
            times=[x['local_total_ms']+8.0*x['wire_bytes']/(mbps*1_000_000)*1000 for x in matches]
            sweep.append({'bandwidth_mbps':mbps,'method':mode,
                          'median_modeled_request_ms':float(np.median(times)),
                          'mean_modeled_request_ms':float(np.mean(times))})
    with (output/'bandwidth_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sweep[0]));w.writeheader();w.writerows(sweep)
    sizes=[{'source_robot':j,'source_snapshot':db.disk_stats(),'geometry':geom.storage()}
           for j,(db,geom) in enumerate(sources)]
    info={'experiment':'BST-ID distributed ROI feasibility v0.1',
          'workload':'synthetic owner-separated robot maps or NCLT versioned-store peer replay',
          'source_robots':len(sources),'requests':requests,'seed':seed,
          'nonempty_requested_region_fraction':nonempty_requests/max(1,requests),
          'nonempty_target_region_fraction':nonempty_targets/max(1,requests),
          'dataset_info':{'dataset':dataset_meta.get('dataset'),
                          'scan_stats':dataset_meta.get('scan_stats'),
                          'extrinsic_mode':dataset_meta.get('extrinsic_mode')} if dataset_meta else None,
          'gps_sigma_horizontal_m':gps_sigma_m,'shared_alignment_sigma_m':alignment_sigma_m,
          'local_map_radius_m':local_radius_m,'request_margin_m':3*math.hypot(gps_sigma_m,alignment_sigma_m),
          'link_bandwidth_mbps':bandwidth_mbps,'bandwidth_is_model_only':True,
          'gps_target_roi_enclosed_fraction':float(np.mean([x['gps_roi_encloses_target'] for x in losses])),
          'gps_actual_voxel_coverage_fraction':float(np.mean([x['source_target_voxel_coverage'] for x in losses])),
          'indexes_built_outside_timed_loop':built_indexes,'storage':sizes,
          'sweep_bandwidth_mbps':[0.25,0.5,1,2,5,10,25],
          'environment':{'python_platform':platform.platform(),'numpy':np.__version__},
          'semantics':'observed endpoint voxels only; unknown != free space; K4 boxes are exact filled cells',
          'validity':'each request recipient reconstruction compared exactly with source ROI',
          'non_claims':['not physical multi-robot SLAM','no real measured network latency',
                        'no scan matching or localization success rate',
                        'geometry savings not solely attributable to BST-ID',
                        'BST and Morton share identical adaptive geometry and same wire codec',
                        'geometry built over full history; no online publication delay benchmarked']}
    (output/'metadata.json').write_text(json.dumps(info,indent=2))
    (output/'report.md').write_text(report_text(summary,info))
    return summary,info


def report_text(summary,info):
    lines=['# BST-ID collaborative map exchange: feasibility screen','',
           'This is **not** a multi-robot SLAM or true localization accuracy benchmark. '
           'It is an exact map-region transfer/caching benchmark with an uncertain GPS request prior.',
           '',f"- Sources: {info['source_robots']}; requests: {info['requests']}; bandwidth model: {info['link_bandwidth_mbps']} Mbps",
           f"- GPS uncertainty: {info['gps_sigma_horizontal_m']} m sigma; shared-map alignment: {info['shared_alignment_sigma_m']} m sigma",
           f"- GPS ROI contains the true localization ROI: {100*info['gps_target_roi_enclosed_fraction']:.1f}% of requests",
           f"- Requests with nonempty sender geometry: {100*info['nonempty_requested_region_fraction']:.1f}% (report zero-overlap requests rather than discard them)",
           '', '| Method | Total bytes | Saving vs versioned voxel tiles | Median local CPU ms | Exact? |',
           '|---|---:|---:|---:|---|']
    for x in summary:
        lines.append(f"| {x['method']} | {x['total_wire_bytes']:,} | {100*x['wire_saving_vs_voxel_tiles']:+.1f}% | {x['median_end_to_end_local_ms']:.2f} | {x['all_correct']} |")
    bst=next(x for x in summary if x['method']=='k4_bst')
    mor=next(x for x in summary if x['method']=='k4_morton')
    lines.extend(['','A `bandwidth_sweep.csv` reports **modeled**, not measured, request latency at 0.25–25 Mbps.',
                  '', '## Interpretation and falsification','',
        '* `voxel_tiles` is a strong, content-addressed baseline; repeated identical tiles require only references. '
        'The ROI protocols cache exact repeated ROI payloads, but do not yet implement partial sub-ROI delta updates.',
        '* K4 adaptive geometry differs from voxel endpoints on the wire, but is exactly reversible for the occupied-voxel endpoint sets. '
        'It does not preserve LiDAR intensity, covariances, free space, or surfels.',
        '* **BST-vs-Morton index comparison** holds adaptive geometry, wire encoding, and receiver constant. '
        f"This run has BST median local round-trip {bst['median_end_to_end_local_ms']:.3f} ms versus Morton {mor['median_end_to_end_local_ms']:.3f} ms; "
        'different CPU timings should be re-tested with more requests and varied regions.',
        '* `k4_tile_morton` and `k4_tile_bst` are strong incremental cached K4-block-tile controls. '
        'They isolate geometry from reusing previous tile transfers.',
        '* A useful result needs an advantage against `voxel_tiles` (bytes and/or end-to-end latency) '
        '**and** no loss of downstream localization. Only the first condition is tested here.',
        '* Online map-frame corrections, pose uncertainty in registration, distributed ownership, actual loss/delay, '
        'and real multi-agent localization remain out of scope.',
        '', '## Decision rule (preliminary)', '',
        'Advance to real NCLT/multi-robot traces only if BST K4 beats efficient voxel tiles by '
        '>=30% in cumulative transmitted bytes at matched exact endpoint geometry, maintains exactness, '
        'and has acceptable sender/receiver CPU cost. Then require downstream real LiDAR matching '
        'to retain >=95% of baseline localization success; this is **not checked** by the feasibility screen.',
        ''])
    return '\n'.join(lines)
