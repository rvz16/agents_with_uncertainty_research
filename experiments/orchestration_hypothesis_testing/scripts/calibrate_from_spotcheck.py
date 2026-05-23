"""Compat shim. Real script lives in calibration/from_spotcheck.py.

Kept so legacy `from calibrate_from_spotcheck import X` calls across the repo
continue to work without modification during the refactor.
Removed in Phase 6 once all callers migrate to
`from calibration.from_spotcheck import ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path so we can import the
# calibration package.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Wildcard re-export of every public symbol from the new location.
# Module-level functions, classes, and constants all carry through.
from calibration.from_spotcheck import *  # noqa: F401, F403, E402

# Also forward any names that don't start with an underscore but that
# wildcard-import would still skip (notably `_make_client`, etc.).
from calibration import from_spotcheck as _new_mod  # noqa: E402
for _name in dir(_new_mod):
    if _name.startswith("__"):
        continue
    if _name not in globals():
        globals()[_name] = getattr(_new_mod, _name)
del _new_mod, _name
