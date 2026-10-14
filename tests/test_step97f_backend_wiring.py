"""Static regression for Step 9.7F optional backend wiring."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_backend_initialization_only_when_required():
    tree = ast.parse((ROOT / "benchmark/localization_benchmark.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark_localization")
    text = ast.unparse(fn)
    assert "if frontend_strategy != 'bev_local' else None" in text
    assert "if local_solver == 'gicp' else None" in text
    assert "if kiss is not None:" in text
    assert "kiss.close()" in text
