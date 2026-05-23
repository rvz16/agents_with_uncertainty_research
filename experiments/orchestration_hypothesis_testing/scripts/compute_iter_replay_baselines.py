"""Compat shim. Real script lives in iter/replay_baselines.py.

Kept so the notebook (analysis.ipynb) and any other legacy callers that
do `from compute_iter_replay_baselines import X` continue to work without
modification through Phase 5. Removed in Phase 6 once those callers
migrate to `from iter.replay_baselines import ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path so we can import the iter package.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Wildcard re-export of every public symbol from the new location.
from iter.replay_baselines import *  # noqa: F401, F403, E402

# Forward underscore-prefixed and other names wildcard-import would skip.
from iter import replay_baselines as _new_mod  # noqa: E402
for _name in dir(_new_mod):
    if _name.startswith("__"):
        continue
    if _name not in globals():
        globals()[_name] = getattr(_new_mod, _name)
del _new_mod, _name
