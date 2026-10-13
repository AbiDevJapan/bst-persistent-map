"""Outputs and frozen metric definitions for the paired map comparison."""
from __future__ import annotations
import statistics
import numpy as np
from benchmark.localization_helpers import _pctl

def _summary(rows):
    out=[]
    for method in sorted(set(r['map_geometry'] for r in rows)):
        g=[r for r in rows if r['map_geometry']==method]
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            'map_geometry':method,'trials':len(g),
            'fine_success_rate':float(np.mean([bool(x['success']) for x in g])),
            'global_2m5deg_success_rate':float(np.mean([bool(x['global_2m5deg_success']) for x in g])),
            'strict_6d_success_rate':float(np.mean([bool(x['strict_6d_success']) for x in g])),
            'median_final_translation_xy_m':statistics.median(vals('final_translation_xy_m')),
            'median_final_translation_3d_m':statistics.median(vals('final_translation_3d_m')),
            'median_final_yaw_error_deg':statistics.median(vals('final_yaw_error_deg')),
            'median_final_rotation_error_deg':statistics.median(vals('final_rotation_error_deg')),
            'p95_final_translation_3d_m':_pctl(vals('final_translation_3d_m'),.95),
            'p95_final_rotation_error_deg':_pctl(vals('final_rotation_error_deg'),.95),
            'median_frontend_ms':statistics.median(vals('frontend_ms')),
            'median_map_retrieval_ms':statistics.median(vals('map_retrieval_ms')),
            'median_local_gicp_ms':statistics.median(vals('local_gicp_ms')),
            'median_total_ms':statistics.median(vals('total_ms')),
            'p95_total_ms':_pctl(vals('total_ms'),.95),
            'mean_local_map_primitives':statistics.mean(vals('local_map_primitives')),
        })
    return out

def _session_summary(rows):
    out=[]
    sessions=sorted(set(r['query_session'] for r in rows))
    methods=sorted(set(r['map_geometry'] for r in rows))
    for s in sessions:
        for m in methods:
            g=[r for r in rows if r['query_session']==s and r['map_geometry']==m]
            if not g: continue
            out.append({
                'query_session':s,'historical_map_session':g[0]['historical_map_session'],'map_geometry':m,'trials':len(g),
                'fine_success_rate':float(np.mean([bool(x['success']) for x in g])),
                'global_2m5deg_success_rate':float(np.mean([bool(x['global_2m5deg_success']) for x in g])),
                'median_translation_xy_m':statistics.median(float(x['final_translation_xy_m']) for x in g),
                'median_yaw_error_deg':statistics.median(float(x['final_yaw_error_deg']) for x in g),
                'median_rotation_error_deg':statistics.median(float(x['final_rotation_error_deg']) for x in g),
                'median_total_ms':statistics.median(float(x['total_ms']) for x in g),
                'mean_local_map_primitives':statistics.mean(float(x['local_map_primitives']) for x in g),
            })
    return out

def _bevplace_summary(frontend_rows):
    if not frontend_rows: return []
    out=[]
    for s in sorted(set(r['query_session'] for r in frontend_rows)) + ['ALL']:
        g=frontend_rows if s=='ALL' else [r for r in frontend_rows if r['query_session']==s]
        if not g: continue
        out.append({
            'query_session':s,'trials':len(g),
            'bevplace_top1_recall_5m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=5 for r in g])),
            'bevplace_top1_recall_10m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=10 for r in g])),
            'bevplace_top1_recall_20m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=20 for r in g])),
            'bevplace_topk_recall_5m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=5 for r in g])),
            'bevplace_topk_recall_10m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=10 for r in g])),
            'bevplace_topk_recall_20m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=20 for r in g])),
            'kiss_candidate_fine_rate':float(np.mean([float(r['frontend_translation_xy_m'])<=0.5 and float(r['frontend_yaw_error_deg'])<=1.0 for r in g])),
            'kiss_candidate_global_2m5deg_rate':float(np.mean([float(r['frontend_translation_3d_m'])<=2.0 and float(r['frontend_rotation_error_deg'])<=5.0 for r in g])),
            'median_frontend_translation_3d_m':statistics.median(float(r['frontend_translation_3d_m']) for r in g),
            'median_frontend_rotation_error_deg':statistics.median(float(r['frontend_rotation_error_deg']) for r in g),
            'median_frontend_ms':statistics.median(float(r['frontend_ms']) for r in g),
        })
    return out

def _scan_context_summary(frontend_rows):
    if not frontend_rows: return []
    out=[]
    for s in sorted(set(r['query_session'] for r in frontend_rows)) + ['ALL']:
        g=frontend_rows if s=='ALL' else [r for r in frontend_rows if r['query_session']==s]
        if not g: continue
        out.append({
            'query_session':s,'trials':len(g),
            'scan_context_top1_recall_5m':float(np.mean([float(r['sc_top1_position_error_m'])<=5 for r in g])),
            'scan_context_top1_recall_10m':float(np.mean([float(r['sc_top1_position_error_m'])<=10 for r in g])),
            'scan_context_top1_recall_20m':float(np.mean([float(r['sc_top1_position_error_m'])<=20 for r in g])),
            'scan_context_topk_recall_5m':float(np.mean([float(r['sc_topk_best_position_error_m'])<=5 for r in g])),
            'scan_context_topk_recall_10m':float(np.mean([float(r['sc_topk_best_position_error_m'])<=10 for r in g])),
            'scan_context_topk_recall_20m':float(np.mean([float(r['sc_topk_best_position_error_m'])<=20 for r in g])),
            'kiss_candidate_fine_rate':float(np.mean([float(r['frontend_translation_xy_m'])<=0.5 and float(r['frontend_yaw_error_deg'])<=1.0 for r in g])),
            'kiss_candidate_global_2m5deg_rate':float(np.mean([float(r['frontend_translation_3d_m'])<=2.0 and float(r['frontend_rotation_error_deg'])<=5.0 for r in g])),
            'median_frontend_translation_3d_m':statistics.median(float(r['frontend_translation_3d_m']) for r in g),
            'median_frontend_rotation_error_deg':statistics.median(float(r['frontend_rotation_error_deg']) for r in g),
            'median_frontend_ms':statistics.median(float(r['frontend_ms']) for r in g),
        })
    return out

