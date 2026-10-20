#!/usr/bin/env python3
"""Download ONLY S3E v1 Playground 2, not the 278 GB S3E repository.

`pip install huggingface_hub` first. Download requires network and ~6.73 GB.
`--ground-truth-only` downloads tiny pose, metadata and calibration files.
"""
from __future__ import annotations
import argparse
from pathlib import Path

REPO='PengYu-Team/S3E'
FOLDER='S3Ev1/S3E_Playground_2'
TINY=('metadata.yaml','alpha_gt.txt','bob_gt.txt','carol_gt.txt')
CALIBRATION=('alpha.yaml','bob.yaml','carol.yaml')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/s3e')
    p.add_argument('--ground-truth-only',action='store_true')
    a=p.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit('Install downloader with: pip install huggingface_hub') from e
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    files=TINY if a.ground_truth_only else TINY+('S3E_Playground_2.db3',)
    names=[f'{FOLDER}/{file}' for file in files]
    names.extend(f'S3Ev1/Calibration/{file}' for file in CALIBRATION)
    for name in names:
        print(f'Downloading {name}',flush=True)
        path=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=name,local_dir=str(folder))
        print('Saved:',path,flush=True)
    print('\nS3E v1 Playground 2 directory:',folder/FOLDER)

if __name__=='__main__':main()
