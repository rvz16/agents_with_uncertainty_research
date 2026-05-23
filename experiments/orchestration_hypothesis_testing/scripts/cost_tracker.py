"""Compatibility shim. Real implementation lives in _common/cost.py.

Kept so legacy `from cost_tracker import CostTracker` imports across the
repo continue working without modification during the refactor.
This shim will be removed in Phase 6 once all callers migrate to
`from _common.cost import CostTracker`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure orchestration_hypothesis_testing/ (parent of scripts/) is on
# sys.path so we can import the _common package.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from _common.cost import (  # noqa: E402, F401
    CostTracker,
    cost_for_call,
    extract_usage,
    project_cost,
)
