from __future__ import annotations

import importlib.metadata
import numpy as np

from localization.step95 import IsolatedKISSMatcherBackend, SmallGICPBackend, pose_errors_6d


def main():
    import small_gicp
    print('kiss-matcher', importlib.metadata.version('kiss-matcher'))
    print('small-gicp', importlib.metadata.version('small-gicp'))

    rng=np.random.default_rng(95)
    src=rng.uniform(-8.0,8.0,size=(5000,3)).astype(np.float64)
    yaw=np.deg2rad(27.0); c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=[4.0,-2.0,0.4]
    tgt=src@R.T+T[:3,3]

    kiss=IsolatedKISSMatcherBackend(0.30)
    kr=kiss.estimate(src,tgt)
    if not kr.converged:
        raise SystemExit(f'KISS-Matcher smoke test failed: {kr.failure_reason}')
    e=pose_errors_6d(kr.T_target_source,T)
    print(f"KISS smoke: inliers={kr.inliers}, error={e['translation_3d_m']:.4f} m / {e['rotation_deg']:.3f} deg")

    gicp=SmallGICPBackend(downsampling_resolution_m=0.30,max_correspondence_distance_m=2.0,num_threads=4,module=small_gicp)
    gr=gicp.align(tgt,src,kr.T_target_source)
    if not gr.converged:
        raise SystemExit(f'small_gicp smoke test failed: {gr.failure_reason}')
    e2=pose_errors_6d(gr.T_target_source,T)
    print(f"GICP smoke: inliers={gr.inliers}, error={e2['translation_3d_m']:.4f} m / {e2['rotation_deg']:.3f} deg")
    kiss.close()
    print('Step 9.5 dependency/API smoke test PASSED (KISS isolated from small_gicp)')


if __name__=='__main__': main()
