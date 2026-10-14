"""Step 9.7A: diagnostic-only analysis of frozen localization runs.

No pose optimization, retriever modification, correspondence re-ranking or GT at runtime.
GT is used only to EVALUATE the localization results.  Existing CSVs yield
absolute-error diagnostics; optional orientation_pose_audit.csv adds signed yaw
and a nearest-neighbor yaw landscape (not the small_gicp Hessian).
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

KEY = ('query_session', 'scan_file', 'original_scan_index')
AUDIT_KEY = ('map_geometry',) + KEY


def _read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def _write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _index(rows, key):
    output = {}
    for item in rows:
        k = tuple(item[field] for field in key)
        if k in output:
            raise ValueError(f'Duplicate query identity: {k}')
        output[k] = item
    return output


def _bool(x):
    if isinstance(x, bool):
        return x
    if str(x).lower() in ('true', '1'):
        return True
    if str(x).lower() in ('false', '0'):
        return False
    raise ValueError(f'Invalid boolean: {x!r}')


def _num(row, name):
    x = float(row[name])
    if not math.isfinite(x):
        raise ValueError(f'Nonfinite {name} at {row.get("scan_file")}: {x}')
    return x


def signed_yaw_error_deg(T_est, T_gt):
    """Signed yaw difference (estimate minus GT) in (-180, 180] degrees."""
    a = np.asarray(T_est, dtype=np.float64)
    b = np.asarray(T_gt, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError('Pose matrices must both be 4x4')
    aa = math.atan2(float(a[1, 0]), float(a[0, 0]))
    bb = math.atan2(float(b[1, 0]), float(b[0, 0]))
    return math.degrees(math.atan2(math.sin(aa-bb), math.cos(aa-bb)))


def yaw_update_deg(T_a, T_b):
    """Signed change from T_a to T_b, world-yaw convention."""
    a = math.atan2(float(T_a[1, 0]), float(T_a[0, 0]))
    b = math.atan2(float(T_b[1, 0]), float(T_b[0, 0]))
    return math.degrees(math.atan2(math.sin(b-a), math.cos(b-a)))


def yaw_landscape(T, source_points, target_points, *, max_corr_m=2.0, max_samples=1500, delta_deg=0.5):
    """Nearest-neighbor *surrogate* yaw objective around a fixed 6D pose.

    Deterministic point sample, yaw around sensor origin, fixed x/y/z and
    roll/pitch. Nearest-neighbor point-to-point squared distances are truncated
    at max_corr_m so rejected matches cannot artificially lower the objective.
    This is NOT small_gicp's GICP Hessian, covariance, or uncertainty estimate.
    If scipy is unavailable, return nulls; do not change registration results.
    """
    fields = ('nn_overlap_fraction', 'nn_rmse_inliers_m',
              'yaw_proxy_objective_center_m2', 'yaw_proxy_curvature_m2_per_rad2',
              'yaw_proxy_preferred_offset_deg', 'yaw_proxy_points')
    blank = {x: None for x in fields}
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return blank
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[1:] != (3,) or tgt.shape[1:] != (3,):
        return blank
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 20 or len(tgt) < 20:
        return blank
    if len(src) > max_samples:
        src = src[np.linspace(0, len(src)-1, max_samples, dtype=np.int64)]
    tree = cKDTree(tgt)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return blank
    h = math.radians(delta_deg)
    costs=[]; hits=[]; rmses=[]
    for yaw in (-h, 0.0, h):
        c, s = math.cos(yaw), math.sin(yaw)
        rz=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)
        world = (src @ rz.T) @ T[:3,:3].T + T[:3,3]
        d,_ = tree.query(world, k=1, distance_upper_bound=max_corr_m, workers=1)
        inlier=np.isfinite(d)
        nd=int(inlier.sum())
        costs.append(float(np.mean(np.where(inlier, d**2, max_corr_m**2))))
        hits.append(float(nd/len(d)))
        rmses.append(float(np.sqrt(np.mean(d[inlier]**2))) if nd else None)
    curvature=float((costs[0]+costs[2]-2*costs[1])/(h*h))
    offset=(-delta_deg,0.0,delta_deg)[int(np.argmin(costs))]
    return dict(nn_overlap_fraction=hits[1], nn_rmse_inliers_m=rmses[1],
                yaw_proxy_objective_center_m2=costs[1],
                yaw_proxy_curvature_m2_per_rad2=curvature,
                yaw_proxy_preferred_offset_deg=offset, yaw_proxy_points=len(src))


def make_pose_audit(*, map_geometry, query_session, scan_file, original_scan_index,
                    T_gt, T_front, T_final, gicp_result, target_points, source_points,
                    max_correspondence_m=2.0, evaluate_landscape=True):
    """Observational logging. T_final must be the ACTUAL pose selected by pipeline."""
    efront=signed_yaw_error_deg(T_front, T_gt)
    efinal=signed_yaw_error_deg(T_final, T_gt)
    trans=np.asarray(T_final[:3,3]-T_front[:3,3],dtype=float)
    data=dict(map_geometry=map_geometry, query_session=query_session,
              scan_file=scan_file, original_scan_index=int(original_scan_index),
              signed_frontend_yaw_error_deg=efront,
              signed_final_yaw_error_deg=efinal,
              signed_map_yaw_update_deg=yaw_update_deg(T_front, T_final),
              abs_yaw_improvement_deg=abs(efront)-abs(efinal),
              map_translation_update_3d_m=float(np.linalg.norm(trans)),
              map_translation_update_xy_m=float(np.linalg.norm(trans[:2])),
              local_gicp_converged=bool(gicp_result.converged),
              local_gicp_error_raw=float(gicp_result.error),
              local_gicp_iterations=int(gicp_result.iterations),
              local_gicp_inliers=int(gicp_result.inliers),
              local_gicp_failure_reason=str(gicp_result.failure_reason),
              local_target_points=int(len(target_points)),
              local_source_points=int(len(source_points)))
    # If GICP did not converge, the current pipeline uses T_front unchanged;
    # we DO NOT treat a non-converged candidate as an accepted pose update.
    data.update(yaw_landscape(T_final,source_points,target_points,max_corr_m=max_correspondence_m)
                if evaluate_landscape else {})
    return data


def analyze(results_dir, output_dir, *, fine_xy_m=0.5, fine_yaw_deg=1.0,
            global_translation_m=2.0, global_rotation_deg=5.0):
    src=Path(results_dir); dst=Path(output_dir); dst.mkdir(parents=True, exist_ok=True)
    loc=_read(src/'localization_trials.csv')
    front=_index(_read(src/'frontend_trials.csv'),KEY)
    if not loc:
        raise ValueError('No localization trials')
    audit_path=src/'orientation_pose_audit.csv'
    audits=_index(_read(audit_path),AUDIT_KEY) if audit_path.is_file() else {}
    extra_columns=[]
    if audits:
        extra_columns=[c for c in next(iter(audits.values())) if c not in AUDIT_KEY]
    enriched=[]
    for r in loc:
        querykey=tuple(r[c] for c in KEY)
        if querykey not in front:
            raise ValueError(f'No frontend row for query {querykey}')
        fr=front[querykey]
        auditkey=(r['map_geometry'],)+querykey
        if audits and auditkey not in audits:
            raise ValueError(f'No pose audit row for {auditkey}')
        fxy=_num(r,'frontend_translation_xy_m'); fyaw=_num(r,'frontend_yaw_error_deg')
        finalxy=_num(r,'final_translation_xy_m'); finalyaw=_num(r,'final_yaw_error_deg')
        frontfine=fxy<=fine_xy_m and fyaw<=fine_yaw_deg
        finalfine=finalxy<=fine_xy_m and finalyaw<=fine_yaw_deg
        if finalfine != _bool(r['success']):
            raise ValueError(f'Existing frozen fine flag is inconsistent: {auditkey}')
        fg=(_num(r,'frontend_translation_3d_m')<=global_translation_m and
            _num(r,'frontend_rotation_error_deg')<=global_rotation_deg)
        hasnear=float(fr.get('bevplace_topk_best_position_error_m',fr.get('sc_topk_best_position_error_m',math.inf)))<=5.0
        if not hasnear:
            category='no_top5_near_reference'
        elif not fg:
            category='near_reference_but_global_pose_failed'
        elif not frontfine:
            category='global_pose_good_but_fine_failed'
        elif not finalfine:
            category='frontend_fine_lost_after_map'
        else:
            category='fine_preserved'
        # Diagnostic categories are observational and mutually exclusive.
        row={**r,
            'frontend_fine_success':frontfine,
            'final_xy_success':finalxy<=fine_xy_m,
            'final_yaw_success':finalyaw<=fine_yaw_deg,
            'frontend_global_success':fg,
            'near_reference_topk_5m':hasnear,
            'fine_transition':('fine_to_fine' if frontfine and finalfine else
                               'fine_to_fail' if frontfine else
                               'fail_to_fine' if finalfine else 'fail_to_fail'),
            'diagnostic_category':category,
            'absolute_yaw_error_change_deg':finalyaw-fyaw,
            'absolute_xy_error_change_m':finalxy-fxy,
            'yaw_only_failure':finalxy<=fine_xy_m and finalyaw>fine_yaw_deg,
            'xy_only_failure':finalxy>fine_xy_m and finalyaw<=fine_yaw_deg,
            'both_fail':finalxy>fine_xy_m and finalyaw>fine_yaw_deg,
            'used_kiss':_bool(fr['kiss_used']),
            'selected_reference_rank':fr['selected_reference_rank'],
            'selected_reference_position_error_m':fr['selected_reference_position_error_m'],
        }
        if audits:
            row.update({c:audits[auditkey][c] for c in extra_columns})
            x=float(row['signed_frontend_yaw_error_deg'])
            y=float(row['signed_final_yaw_error_deg'])
            if abs(abs(x)-fyaw)>1e-3 or abs(abs(y)-finalyaw)>1e-3:
                raise ValueError(f'Signed yaw audit disagrees with frozen CSV: {auditkey}')
            if not _bool(r['local_gicp_converged']) and (abs(float(row['signed_map_yaw_update_deg']))>1e-6 or abs(float(row['map_translation_update_3d_m']))>1e-6):
                raise ValueError(f'Unexpected non-converged pose update: {auditkey}')
        enriched.append(row)
    _write(dst/'orientation_trials.csv',enriched)
    summary=[]
    for method in sorted(set(r['map_geometry'] for r in enriched)):
        group=[r for r in enriched if r['map_geometry']==method]
        for session in sorted(set(r['query_session'] for r in group))+['ALL']:
            g=group if session=='ALL' else [r for r in group if r['query_session']==session]
            count=lambda field:sum(_bool(r[field]) for r in g)
            f=Counter(r['fine_transition'] for r in g)
            conv=[r for r in g if _bool(r['local_gicp_converged'])]
            summary.append(dict(map_geometry=method,query_session=session,queries=len(g),
                frontend_fine=count('frontend_fine_success'),final_fine=sum(_bool(r['success']) for r in g),
                final_xy=count('final_xy_success'),final_yaw=count('final_yaw_success'),
                yaw_only_failures=count('yaw_only_failure'),xy_only_failures=count('xy_only_failure'),
                both_failures=count('both_fail'),
                front_fine_lost=f['fine_to_fail'],front_not_fine_recovered=f['fail_to_fine'],
                converged=len(conv),converged_fine=sum(_bool(r['success']) for r in conv),
                no_convergence_fine=sum(_bool(r['success']) for r in g if not _bool(r['local_gicp_converged'])),
                frontend_global=count('frontend_global_success'),
                topk_near_reference=count('near_reference_topk_5m'),
                median_final_yaw_error_deg=float(np.median([float(r['final_yaw_error_deg']) for r in g])),
                median_abs_yaw_error_magnitude_change_deg=float(np.median([abs(float(r['absolute_yaw_error_change_deg'])) for r in g]))))
            if audits:
                summary[-1].update(
                    mean_signed_frontend_yaw_deg=float(np.mean([float(r['signed_frontend_yaw_error_deg']) for r in g])),
                    mean_signed_final_yaw_deg=float(np.mean([float(r['signed_final_yaw_error_deg']) for r in g])),
                    median_signed_final_yaw_deg=float(np.median([float(r['signed_final_yaw_error_deg']) for r in g])),
                    median_signed_map_update_deg=float(np.median([float(r['signed_map_yaw_update_deg']) for r in g])),
                    median_map_translation_update_xy_m=float(np.median([float(r['map_translation_update_xy_m']) for r in g])))
    _write(dst/'orientation_summary.csv',summary)
    cats=[]
    for method in sorted(set(r['map_geometry'] for r in enriched)):
        for session in sorted(set(r['query_session'] for r in enriched))+['ALL']:
            g=[r for r in enriched if r['map_geometry']==method and (session=='ALL' or r['query_session']==session)]
            for category,n in sorted(Counter(x['diagnostic_category'] for x in g).items()):
                cats.append(dict(map_geometry=method,query_session=session,diagnostic_category=category,queries=n))
    _write(dst/'failure_categories.csv',cats)
    if audits:
        def quality_bucket(x):
            if not x or x.lower() in ('none','nan'):
                return 'not_measured'
            z=float(x)
            if not math.isfinite(z):
                return 'not_measured'
            return 'positive_proxy_curvature' if z>0 else 'flat_or_negative_proxy_curvature'
        quality=[]
        for method in sorted(set(x['map_geometry'] for x in enriched)):
            g=[x for x in enriched if x['map_geometry']==method]
            bins=sorted(set((quality_bucket(x.get('yaw_proxy_curvature_m2_per_rad2','')),
                             _bool(x['local_gicp_converged'])) for x in g))
            for curvature,converged in bins:
                sub=[x for x in g if quality_bucket(x.get('yaw_proxy_curvature_m2_per_rad2',''))==curvature
                     and _bool(x['local_gicp_converged'])==converged]
                quality.append(dict(map_geometry=method,curvature_proxy_bucket=curvature,
                    local_gicp_converged=converged,queries=len(sub),
                    fine_success=sum(_bool(x['success']) for x in sub),
                    yaw_only_failures=sum(_bool(x['yaw_only_failure']) for x in sub),
                    median_nn_overlap_fraction=(float(np.median([float(x['nn_overlap_fraction']) for x in sub
                         if x.get('nn_overlap_fraction','').lower() not in ('','none','nan')]))
                         if any(x.get('nn_overlap_fraction','').lower() not in ('','none','nan') for x in sub) else '')))
        _write(dst/'orientation_quality.csv',quality)
    # Confirm the paired frontend is identical for both methods, as required.
    by_method={m:_index([r for r in enriched if r['map_geometry']==m],KEY) for m in sorted(set(r['map_geometry'] for r in enriched))}
    if len(by_method)==2:
        a,b=list(by_method.values())
        if set(a)!=set(b):
            raise ValueError('Voxel and BST queried different scan identities')
        for k in a:
            if (abs(float(a[k]['frontend_yaw_error_deg'])-float(b[k]['frontend_yaw_error_deg']))>1e-6 or
                abs(float(a[k]['frontend_translation_xy_m'])-float(b[k]['frontend_translation_xy_m']))>1e-6):
                raise ValueError(f'Different frontend pose for same query: {k}')
    note={'step':'9.7A','input':str(src),'trials':len(enriched),'query_count':len(front),
          'has_signed_yaw_audit':bool(audits),'fine_xy_m':fine_xy_m,'fine_yaw_deg':fine_yaw_deg,
          'diagnostic_only':True,
          'warning':'Absolute yaw errors cannot reveal yaw sign or physical pose update. '
                    'For signed yaw and nearest-neighbor surrogate observability rerun the unified '
                    'benchmark with --orientation-diagnostics. The proxy is NOT the small_gicp Hessian. '
                    'The no_top5_near_reference category does not distinguish absent database coverage from retrieval error. Failure categories are descriptive, not causal proof.'}
    (dst/'orientation_metadata.json').write_text(json.dumps(note,indent=2))
    return summary
