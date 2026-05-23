"""Compat shim. Real script lives in analysis/controller.py.

Kept so the notebook (analysis.ipynb) and other legacy callers that do
`from run_baseline_vs_controller import X` continue to work without modification through the
refactor. Removed in Phase 6 once those callers migrate to
`from analysis.controller import ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path so we can import the analysis package.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from analysis.controller import *  # noqa: F401, F403, E402

# Forward names wildcard-import skips (underscored, etc.)
from analysis import controller as _new_mod  # noqa: E402
for _name in dir(_new_mod):
    if _name.startswith("__"):
        continue
    if _name not in globals():
        globals()[_name] = getattr(_new_mod, _name)
del _new_mod, _name
