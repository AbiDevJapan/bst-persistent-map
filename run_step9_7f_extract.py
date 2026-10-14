"""Extract FAST-keypoint BEVPlace++ REIN features using the Step 9.6 checkpoint.

This is a separate offline reference-feature build + query extractor. The
benchmark times feature matching only; online query extraction must be timed
separately for a complete sensor-to-pose latency claim.
"""
import argparse
from pathlib import Path
import numpy as np
from run_step9_6_extract import _model,_read_body,bev_image,PREPROCESS_ID,_sha256
from realdata.nclt import NCLTIngestConfig,discover_nclt_session,_list_scan_files

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--descriptor-cache-dir',required=True)
    p.add_argument('--bevplace-repo',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=('mps','cpu'),default='cpu')
    p.add_argument('--max-keypoints',type=int,default=512)
    a=p.parse_args()
    import torch,cv2,open3d as o3d
    model,_=_model(a.bevplace_repo,Path(a.checkpoint),torch.device(a.device))
    cfg=NCLTIngestConfig(voxel_size_m=0.5,scan_stride=10,min_range_m=2.0,max_range_m=80.0,max_pose_gap_us=100000)
    descriptor=Path(a.descriptor_cache_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    expected_hash=_sha256(a.checkpoint)
    for file in sorted(descriptor.glob('*.npz')):
        if not (file.name.startswith('reference_') or file.name.startswith('query_')):continue
        with np.load(file,allow_pickle=False) as z:
            session=file.stem.split('_',1)[1]
            kind=file.stem.split('_',1)[0]
            names=z['scan_files'].astype(str);indices=z['original_scan_indices'].astype(int)
            assert str(z['checkpoint_sha256'])==expected_hash,'Checkpoint mismatch with Step9.6'
            assert str(z['preprocess_id'])==PREPROCESS_ID,'BEV preprocessing mismatch'
        raw=_list_scan_files(discover_nclt_session(a.data_root,session).scan_dir)
        offsets=[0];xy=[];feat=[];detector=cv2.FastFeatureDetector_create()
        print(f'[{kind} {session}] {len(names)} scans',flush=True)
        for i,(name,index) in enumerate(zip(names,indices)):
            path=raw[int(index)]
            if path.name!=name:raise ValueError(f'Filename mismatch {path.name} != {name}')
            img=bev_image(_read_body(path,cfg),o3d)
            kps=sorted(detector.detect(img),key=lambda k:-k.response)[:a.max_keypoints]
            tensor=np.repeat(img[None,:,:],3,axis=0).astype(np.float32)/256.0
            with torch.inference_mode():
                _,local,_=model(torch.from_numpy(tensor[None]).to(a.device))
            local=local[0].detach().float().cpu().numpy()
            # Never silently resize features: official matching assumes pixel alignment.
            if local.shape[1:]!=img.shape:
                raise ValueError(f'Upstream feature resolution {local.shape[1:]} != BEV {img.shape}; add explicitly verified coordinate mapping')
            coords=np.array([k.pt for k in kps],dtype=float).reshape(-1,2)
            col=np.clip(np.rint(coords[:,0]).astype(int),0,200)
            row=np.clip(np.rint(coords[:,1]).astype(int),0,200)
            xy.append(np.column_stack(((100-row)*0.4,(100-col)*0.4)).astype(np.float32))
            feat.append(local[:,row,col].T.astype(np.float16))
            offsets.append(offsets[-1]+len(kps))
            if (i+1)%250==0:print(f'  {i+1}/{len(names)}',flush=True)
        np.savez_compressed(out/f'{kind}_{session}_local.npz',scan_files=names,original_scan_indices=indices,
            offsets=np.array(offsets,dtype=np.int64),xy=np.concatenate(xy),features=np.concatenate(feat),
            preprocess_id=PREPROCESS_ID,checkpoint_sha256=expected_hash)
if __name__=='__main__':main()
