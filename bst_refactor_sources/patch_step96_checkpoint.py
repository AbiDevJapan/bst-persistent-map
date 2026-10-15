#!/usr/bin/env python3
"""Safely update the Step 9.6 MPS extractor to load a legacy BEVPlace++ checkpoint.

Keeps weights_only=True, allowlists only NumPy scalar/dtype serialization
helpers, backs up the extractor, and never modifies the upstream model or
checkpoint. Apply from the BST project root:
    python patch_step96_checkpoint.py
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

OLD = '''    try:
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
    except TypeError:
        # Old PyTorch without weights_only parameter: only load trusted files.
        state=torch.load(str(checkpoint),map_location='cpu')
'''
NEW = '''    # This upstream checkpoint stores some metadata as NumPy scalars.
    # Keep the restricted weights-only unpickler; explicitly allowlist only
    # the NumPy serialization helpers needed by trusted legacy checkpoints.
    # The tuple preserves the historical pickle path under NumPy 2.x.
    from numpy.core.multiarray import scalar as _numpy_scalar
    with torch.serialization.safe_globals([
        (_numpy_scalar, 'numpy.core.multiarray.scalar'),
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
    ]):
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extractor',default='run_step9_6_extract.py')
    a=p.parse_args()
    path=Path(a.extractor).resolve()
    source=path.read_text(encoding='utf-8')
    if source.count(OLD)!=1:
        raise RuntimeError('Expected original Step 9.6 torch.load block once; file differs or has already been patched. No changes made.')
    if '--device' not in source:
        raise RuntimeError('MPS patch not detected: apply patch_step96_m3.py first. No changes made.')
    modified=source.replace(OLD,NEW)
    ast.parse(modified,filename=str(path))
    backup=path.with_name(path.name+'.pre_checkpoint_patch')
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}; no changes made.')
    backup.write_text(source,encoding='utf-8')
    path.write_text(modified,encoding='utf-8')
    print('Patched:',path)
    print('Backup:',backup)
    print('PyTorch weights_only=True preserved; official model and checkpoint unchanged.')

if __name__=='__main__':
    main()
