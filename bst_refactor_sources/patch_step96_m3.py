#!/usr/bin/env python3
"""Adapt the Step 9.6 BEVPlace++ descriptor extractor for Apple M3/MPS.

Non-destructive with respect to upstream: does not edit the official REIN.py.
Makes a portable copy of REIN.py beside the extractor, and backs up the
original extractor before changing its device handling. Use a *trusted*
pretrained BEVPlace++ checkpoint and validate descriptor equivalence separately.

Example, from BST-ID project root:
    python patch_step96_m3.py --repo third_party/BEVPlace2
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def replace_exact(source: str, old: str, new: str, count: int = 1, label: str = "") -> str:
    found = source.count(old)
    if found != count:
        raise RuntimeError(
            f"Cannot apply {label or old[:70]!r}: expected {count} match(es), found {found}. "
            "This script targets the Step 9.6 extractor supplied in this conversation; "
            "do not overwrite the local version if it has diverged."
        )
    return source.replace(old, new)


def produce_patch(extractor: Path, upstream: Path):
    official = upstream / "REIN.py"
    if not official.is_file():
        raise FileNotFoundError(f"Official upstream REIN.py not found: {official}")
    if not extractor.is_file():
        raise FileNotFoundError(f"Step 9.6 extractor not found: {extractor}")

    # Prepare everything in memory before changing any file.
    rein = official.read_text(encoding="utf-8")
    rein = replace_exact(
        rein,
        "self.angles = -torch.arange(0,359.00001,360.0/rotations)/180*torch.pi",
        "self.register_buffer('angles', -torch.arange(0,359.00001,360.0/rotations)/180*torch.pi, persistent=False)",
        label="rotation angles into portable non-persistent buffer",
    )
    rein = replace_exact(
        rein,
        "torch.zeros(batch_size,2,3).cuda()",
        "torch.zeros((batch_size, 2, 3), device=x.device, dtype=x.dtype)",
        count=3,
        label="all three hard-coded CUDA affine matrices",
    )

    code = extractor.read_text(encoding="utf-8")
    if "--device" in code or "_bevplace_REIN_mps.py" in code:
        raise RuntimeError("Extractor already appears MPS-patched; refusing a second patch")
    code = replace_exact(code,
        "    if not torch.cuda.is_available():\n        raise RuntimeError('Authors\\' unmodified REIN.py uses hard-coded .cuda(); run this extractor on CUDA/Linux. Precomputed descriptor NPZs can be used on macOS.')\n",
        "    # The model copy uses the requested device; official source remains unchanged.\n",
        label="remove extractor CUDA-only guard")
    code = replace_exact(code, "def _model(repo, checkpoint):", "def _model(repo, checkpoint, device):", label="model signature")
    code = replace_exact(code,
        "spec=importlib.util.spec_from_file_location('bevplace2_author_REIN',repo/'REIN.py')",
        "spec=importlib.util.spec_from_file_location('bevplace2_author_REIN_mps',Path(__file__).with_name('_bevplace_REIN_mps.py'))",
        label="load portable model copy")
    code = replace_exact(code, "model=model.cuda().eval()", "model=model.to(device).eval()", label="model transfer")
    code = replace_exact(code,
        "def _encode_scan_files(paths, model, torch, cfg, o3d, batch_size):",
        "def _encode_scan_files(paths, model, torch, cfg, o3d, batch_size, device):",
        label="encoder signature")
    code = replace_exact(code,
        "x=torch.from_numpy(np.stack(pending)).cuda()",
        "x=torch.from_numpy(np.stack(pending)).to(device)",
        label="batch transfer")
    code = replace_exact(code,
        "torch.cuda.synchronize(); t0=time.perf_counter()",
        "if device.type == 'mps': torch.mps.synchronize()\n            t0=time.perf_counter()",
        label="before-network MPS synchronization")
    code = replace_exact(code,
        "torch.cuda.synchronize(); durations.append(1000*(time.perf_counter()-t0))",
        "if device.type == 'mps': torch.mps.synchronize()\n            durations.append(1000*(time.perf_counter()-t0))",
        label="after-network MPS synchronization")
    code = replace_exact(code,
        "    p.add_argument('--batch-size',type=int,default=4)",
        "    p.add_argument('--batch-size',type=int,default=1)\n"
        "    p.add_argument('--device',choices=('auto','mps','cpu'),default='auto',help='M3: use mps; CPU for fallback/debugging')",
        label="device CLI + safe batch default")
    code = replace_exact(code,
        "    model,torch=_model(a.bevplace_repo,checkpoint)",
        "    import torch\n"
        "    if a.device == 'mps' and not torch.backends.mps.is_available():\n"
        "        raise RuntimeError('MPS is unavailable: check PyTorch macOS arm64 installation')\n"
        "    chosen = 'mps' if a.device == 'auto' and torch.backends.mps.is_available() else ('cpu' if a.device == 'auto' else a.device)\n"
        "    device = torch.device(chosen)\n"
        "    print(f'[BEVPlace++] device={device}, batch_size={a.batch_size}', flush=True)\n"
        "    model,torch=_model(a.bevplace_repo,checkpoint,device)",
        label="device selection in main")
    code = replace_exact(code,
        "vec,network_ms=_encode_scan_files(files,model,torch,cfg,o3d,a.batch_size)",
        "vec,network_ms=_encode_scan_files(files,model,torch,cfg,o3d,a.batch_size,device)",
        label="pass device to inference")
    code = code.replace(
        "The authors' REIN.py currently calls .cuda() directly, so CUDA is required.",
        "A locally generated non-CUDA REIN model copy supports MPS and CPU.",
    )
    # Preserve backend provenance separately from descriptor preprocessing/model ID.
    if "per_session=metadata),indent=2))" in code:
        code = replace_exact(
            code, "per_session=metadata),indent=2))",
            "per_session=metadata,device=str(device),torch_version=str(torch.__version__)),indent=2))",
            label="record inference backend for reproducibility",
        )
    ast.parse(rein)
    ast.parse(code)
    portable = extractor.with_name("_bevplace_REIN_mps.py")
    backup = extractor.with_suffix(extractor.suffix + ".pre_mps_backup")
    if backup.exists() or portable.exists():
        raise FileExistsError(f"Existing backup or portable model found: {backup} or {portable}; refusing overwrite")
    backup.write_text(extractor.read_text(encoding='utf-8'), encoding='utf-8')
    portable.write_text(rein, encoding='utf-8')
    extractor.write_text(code, encoding='utf-8')
    return backup, portable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extractor', default='run_step9_6_extract.py')
    parser.add_argument('--repo', default='third_party/BEVPlace2')
    args = parser.parse_args()
    backup, portable = produce_patch(Path(args.extractor).resolve(), Path(args.repo).resolve())
    print('Patched extractor for Apple MPS / CPU.')
    print('Backup:', backup)
    print('Portable REIN model:', portable)
    print('Official REIN.py remains unchanged.')


if __name__ == '__main__':
    main()
