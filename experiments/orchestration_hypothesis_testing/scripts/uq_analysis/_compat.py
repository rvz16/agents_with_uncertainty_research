"""Import shim: the analysis scripts were written against the `src/code_uq`
extraction, this branch keeps the live-runner layout.

The two lineages hold the same modules under different names — the extraction
packages them (`code_uq.analysis.X`, `trajectory_uq_toolkit.X`), the live runner
keeps them flat under `scripts/`. Rather than rewriting sixteen scripts, this
module registers the aliases once so both spellings resolve.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
ORCH = SCRIPTS.parent
REPO = ORCH.parent.parent

for path in (str(REPO), str(ORCH), str(SCRIPTS), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _alias(package: str, module: str, target: str) -> None:
    """Expose `target` (importable now) as `package.module`."""
    pkg = sys.modules.get(package)
    if pkg is None:
        pkg = types.ModuleType(package)
        pkg.__path__ = []          # namespace-like, so submodules can attach
        sys.modules[package] = pkg
    loaded = importlib.import_module(target)
    sys.modules[f"{package}.{module}"] = loaded
    setattr(pkg, module.rsplit(".", 1)[-1], loaded)


def install() -> None:
    _alias("code_uq", "analysis", "types")          # placeholder parent
    sys.modules["code_uq.analysis"] = types.ModuleType("code_uq.analysis")
    sys.modules["code_uq.analysis"].__path__ = []
    setattr(sys.modules["code_uq"], "analysis", sys.modules["code_uq.analysis"])

    for name, target in (
        ("analyze_lcb_llm_tool_agent_logs", "analyze_lcb_llm_tool_agent_logs"),
        # the student branch's copy: it is a superset (486 lines vs 222) and adds
        # the continuous / tempered Gaussian fusion the analysis depends on,
        # which this branch's scripts/experiment2_uq_bayes_critic.py lacks.
        ("experiment2_uq_bayes_critic", "experiment2_uq_bayes_critic_full"),
        ("uq_features", "uq_features"),
        ("bayes_trajectory", "bayes_trajectory"),
        ("entropy_kl_trajectory", "entropy_kl_trajectory"),
    ):
        loaded = importlib.import_module(target)
        sys.modules[f"code_uq.analysis.{name}"] = loaded
        setattr(sys.modules["code_uq.analysis"], name, loaded)

    toolkit = types.ModuleType("trajectory_uq_toolkit")
    toolkit.__path__ = []
    sys.modules["trajectory_uq_toolkit"] = toolkit
    for name, target in (("bayes", "toolkit_bayes"), ("metrics", "toolkit_metrics")):
        loaded = importlib.import_module(target)
        sys.modules[f"trajectory_uq_toolkit.{name}"] = loaded
        setattr(toolkit, name, loaded)


install()
