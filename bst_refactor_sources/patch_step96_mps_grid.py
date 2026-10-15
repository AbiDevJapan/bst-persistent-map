#!/usr/bin/env python3
"""Remove BEVPlace++ legacy .type(x.type()) grid casts, invalid on PyTorch MPS.

Applies ONLY to the local Apple-compatible _bevplace_REIN_mps.py, NEVER to
third_party/BEVPlace2/REIN.py. Checks four expected occurrences before changing.
Creates a backup and validates resulting Python syntax.

Usage (from bst-persistent-map project root):
    python patch_step96_mps_grid.py
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def patch_model(path: Path) -> tuple[Path, int]:
    path = Path(path).resolve()
    if path.name != '_bevplace_REIN_mps.py':
        raise ValueError(f'Refusing to modify non-portable model file: {path}')
    if not path.is_file():
        raise FileNotFoundError(f'MPS model copy not found: {path}. Run the earlier M3 patch first.')
    before = path.read_text(encoding='utf-8')
    needle = '.type(x.type())'
    count = before.count(needle)
    if count != 4:
        raise RuntimeError(
            f'Expected 4 BEVPlace++ legacy grid casts, found {count}. '
            'File may differ from the official REIN model or already be patched; not editing it.'
        )
    # In the previous M3 patch, each affine matrix already uses
    # device=x.device and dtype=x.dtype. affine_grid inherits both, so the
    # four extra tensor.type(string) casts are redundant. Removing them avoids
    # the unsupported "torch.mps.FloatTensor" type string.
    updated = before.replace(needle, '')
    ast.parse(updated, filename=str(path))
    backup = path.with_name(path.name + '.pre_grid_patch')
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}; refusing overwrite.')
    backup.write_text(before, encoding='utf-8')
    path.write_text(updated, encoding='utf-8')
    return backup, count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='_bevplace_REIN_mps.py')
    args = p.parse_args()
    backup, count = patch_model(Path(args.model))
    print(f'Updated {count} legacy affine_grid casts in {Path(args.model).resolve()}')
    print(f'Original model copy backed up at {backup}')
    print('Official BEVPlace2/REIN.py and checkpoint were not modified.')


if __name__ == '__main__':
    main()
