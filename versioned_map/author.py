"""Load the author-supplied package under a private name to avoid collisions
with the project's pre-existing ``bst_id`` prototype package. Vendored source
is unmodified. This is a normal Python package loaded from the archive.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ALIAS = '_step10_bstid_author'

def author_module(relative_name):
    if ALIAS not in sys.modules:
        root = Path(__file__).resolve().parents[1] / 'third_party' / 'bstid_author' / 'bst_id'
        spec = importlib.util.spec_from_file_location(
            ALIAS, root / '__init__.py',
            submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None:
            raise ImportError('Author BST-ID package is missing')
        package = importlib.util.module_from_spec(spec)
        sys.modules[ALIAS] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(ALIAS, None)
            raise
    return __import__(ALIAS + '.' + relative_name, fromlist=['*'])
