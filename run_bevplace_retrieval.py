"""Fast, causal BEVPlace++ retrieval-only diagnostic before costly KISS/BST rebuild.

Runs on CPU from cached descriptor NPZs. Ground truth is evaluation-only.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from localization.bevplace import BEVPlaceDescriptorCache
from realdata.nclt import discover_nclt_session, load_ground_truth, _list_scan_files, _timestamp_from_scan
from realdata.nclt_packed import _nearest_pose_rows


def _write_csv(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate(data_root, query_scans_csv, descriptor_cache_dir, output_dir, topks=(1,5,10,20)):
    rows=list(csv.DictReader(Path(query_scans_csv).open(newline='')))
    if not rows: raise ValueError('Empty query scans file')
    if any('historical_map_session' not in r for r in rows):
        raise ValueError('Expected query_scans.csv containing historical_map_session')
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    cache=BEVPlaceDescriptorCache(descriptor_cache_dir)
    retrieval=[]; metadata=[]
    for session in list(dict.fromkeys(r['query_session'] for r in rows)):
        subset=[r for r in rows if r['query_session']==session and r.get('scenario','cross_session')=='cross_session']
        if not subset:continue
        prior={r['historical_map_session'] for r in subset}
        if len(prior)!=1: raise ValueError(f'{session}: more than one historical database in query CSV')
        prior=prior.pop()
        if prior>=session: raise ValueError('Causal prior session must predate query session')
        with np.load(Path(descriptor_cache_dir)/f'reference_{prior}.npz',allow_pickle=False) as z:
            expected=z['original_scan_indices'].copy(); files=z['scan_files'].copy()
        db,pair=cache.load_pair(prior,session,expected,files)
        if len(db.descriptors)<max(topks):
            raise ValueError('Reference database smaller than requested maximum top-K')
        with np.load(Path(descriptor_cache_dir)/f'reference_{prior}.npz',allow_pickle=False) as z:
            ref_pos=np.asarray(z['pose_rows'],dtype=np.float64)[:,1:3]
        paths=discover_nclt_session(data_root,session)
        scans=_list_scan_files(paths.scan_dir)
        gt=load_ground_truth(paths.gt_csv)
        indices=np.asarray([int(r['original_scan_index']) for r in subset],dtype=np.int64)
        names=[str(r['scan_file']) for r in subset]
        if any(scans[int(i)].name!=name for i,name in zip(indices,names)):
            raise ValueError('Ground-truth scan index/name mismatch')
        stamps=np.array([_timestamp_from_scan(scans[int(i)]) for i in indices],dtype=np.int64)
        poses,gaps=_nearest_pose_rows(gt,stamps)
        if np.any(gaps>100000) or not np.all(np.isfinite(poses[:,1:7])):
            raise ValueError('Query GT poses invalid for frozen 100ms NCLT max gap')
        for k,(oi,name,pose) in enumerate(zip(indices,names,poses)):
            q=cache.query_descriptor(pair['query_by_name'],name,int(oi))
            matches=db.query_descriptor(q,top_k=max(topks))
            cand=np.asarray([m.reference_index for m in matches],dtype=np.int64)
            dists=np.linalg.norm(ref_pos-pose[None,1:3],axis=1)
            oracle=float(np.min(dists))
            rec=dict(query_session=session,historical_map_session=prior,scan_file=name,original_scan_index=int(oi),
                     oracle_reference_distance_xy_m=oracle,db_reference_count=len(ref_pos),
                     top1_distance_xy_m=float(dists[cand[0]]),top1_descriptor_distance=matches[0].distance)
            for topk in topks:
                closest=float(np.min(dists[cand[:int(topk)]]))
                rec[f'best_top{topk}_distance_xy_m']=closest
                for th in (5,10,20): rec[f'recall_top{topk}_{th}m']=int(closest<=th)
            for th in (5,10,20): rec[f'oracle_coverage_{th}m']=int(oracle<=th)
            retrieval.append(rec)
        metadata.append(dict(query_session=session,historical_map_session=prior,queries=len(subset),references=len(ref_pos)))
    _write_csv(root/'retrieval_trials.csv',retrieval)
    summaries=[]
    for session in list(dict.fromkeys(r['query_session'] for r in retrieval))+['ALL']:
        grp=retrieval if session=='ALL' else [r for r in retrieval if r['query_session']==session]
        if not grp:continue
        rec=dict(query_session=session,queries=len(grp))
        for th in (5,10,20):rec[f'oracle_coverage_{th}m']=float(np.mean([r[f'oracle_coverage_{th}m'] for r in grp]))
        for topk in topks:
            for th in (5,10,20):
                rec[f'recall_top{topk}_{th}m']=float(np.mean([r[f'recall_top{topk}_{th}m'] for r in grp]))
        summaries.append(rec)
    _write_csv(root/'retrieval_summary.csv',summaries)
    (root/'metadata.json').write_text(json.dumps(dict(step='9.6_retrieval_only',descriptor_cache_dir=str(descriptor_cache_dir),
        evaluation_protocol='Frozen Step 9.5 paired queries, exact causal latest-prior reference keyframes, no GT in retrieval',
        topks=list(topks),sessions=metadata),indent=2))
    row=summaries[-1]
    print(f"{row['queries']} queries, oracle@5m={100*row['oracle_coverage_5m']:.1f}%",flush=True)
    for k in topks:print(f"  Recall@{k} within 5m: {100*row[f'recall_top{k}_5m']:.1f}%",flush=True)
    print(f'Wrote {root}',flush=True)
    return summaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--query-scans-csv',required=True)
    p.add_argument('--descriptor-cache-dir',required=True)
    p.add_argument('--output',default='results/step9_6_retrieval')
    a=p.parse_args()
    evaluate(a.data_root,a.query_scans_csv,a.descriptor_cache_dir,a.output)

if __name__=='__main__':main()
