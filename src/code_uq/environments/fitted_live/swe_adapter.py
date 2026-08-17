"""SWE-Bench Lite/Verified adapter for live fitted-controller runs."""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .common import Candidate, CriticResult, VerifyResult, feedback_block

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def harness_status() -> tuple[bool, str]:
    """Whether the official SWE-bench harness can actually produce a label.

    The harness builds and runs a container per instance, so it needs both the
    ``swebench`` package and a container runtime. Checked once, up front: the
    alternative is discovering it per instance, where each failure looks
    exactly like a patch that did not fix the bug.
    """
    if find_spec("swebench") is None:
        return False, "the swebench package is not installed"
    runtime = next(
        (name for name in ("docker", "podman", "nerdctl") if shutil.which(name)), None
    )
    if runtime is None:
        return False, "no container runtime found (docker/podman/nerdctl)"
    return True, f"swebench harness with {runtime}"


@dataclass
class SWEAdapter:
    benchmark: str
    dataset_name: str
    n_instances: int
    seed: int
    output_dir: Path
    harness_workers: int = 1
    _oracle_cache: dict[str, dict[str, str]] = field(default_factory=dict)

    def load_instances(self) -> list[dict]:
        from code_uq.scripts import spot_check_generators as scg

        return scg.sample_instances(
            seed=self.seed,
            n=self.n_instances,
            dataset_name=self.dataset_name,
        )

    def instance_id(self, instance: dict) -> str:
        return str(instance["instance_id"])

    def _oracle_files(self, instance: dict) -> dict[str, str]:
        """Contents of the files the gold patch touches.

        Read from the on-disk cache written by ``scripts/prefetch_data.py``
        when ``SWE_ORACLE_CACHE`` is set. Falling back to fetching them from
        raw.githubusercontent works, but only on a host that still has network
        access -- which an offline run is not supposed to have.
        """
        from code_uq.scripts import spot_check_generators as scg

        inst_id = self.instance_id(instance)
        if inst_id in self._oracle_cache:
            return self._oracle_cache[inst_id]

        cache_dir = os.environ.get("SWE_ORACLE_CACHE", "").strip()
        if cache_dir:
            cached = Path(cache_dir) / f"{inst_id}.json"
            if cached.exists():
                self._oracle_cache[inst_id] = json.loads(cached.read_text())
                return self._oracle_cache[inst_id]
            log.warning(
                "no cached oracle files for %s in %s; falling back to the network",
                inst_id,
                cache_dir,
            )

        files = scg.get_changed_files_from_patch(instance.get("patch", "") or "")
        self._oracle_cache[inst_id] = scg.fetch_oracle_files(
            instance["repo"],
            instance["base_commit"],
            files,
        )
        return self._oracle_cache[inst_id]

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from code_uq.scripts import spot_check_generators as scg

        return scg.make_prompt(instance, self._oracle_files(instance)) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from code_uq.scripts import spot_check_generators as scg

        oracle = self._oracle_files(instance)
        diff, extraction_path, n_blocks = scg._extract_diff_from_response(response_text, oracle)
        return Candidate(
            payload=diff,
            raw_text=response_text,
            kind="diff",
            metadata={"extraction_path": extraction_path, "n_blocks": n_blocks},
        )

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        if critic == "L2":
            # SWE-Bench has no public-test critic in this pipeline. Reported as
            # "no verdict" rather than FAIL so a critic that never ran does not
            # look like a critic that rejected the patch.
            return CriticResult(None, detail="unsupported_for_swe")

        from code_uq.environments.calibration import from_spotcheck as cfs

        diff = candidate.payload or ""
        if not diff.strip():
            return CriticResult(False, detail="empty_diff")
        if critic == "L3":
            ok, cost = cfs.critic_L3_llm_review(
                self.instance_id(instance),
                instance.get("problem_statement", ""),
                diff,
                reviewer_client,
            )
            return CriticResult(
                None if ok is None else bool(ok),
                detail="" if ok is not None else "reviewer_unavailable",
                api_cost_usd=float(cost),
            )

        modified = cfs._modified_file_contents(diff, self._oracle_files(instance))
        if modified is None:
            return CriticResult(False, detail="diff_apply_failed")
        if critic == "L0":
            return CriticResult(bool(cfs.critic_L0_syntax(modified)))
        if critic == "L1":
            return CriticResult(bool(cfs.critic_L1_lint(modified)))
        raise ValueError(f"unknown critic: {critic}")

    def unavailable_actions(self) -> set[str]:
        """Actions the controller should not be offered a step to spend.

        Cost is the lesser reason. The real one: a "failure" reported by an
        action that never ran must not enter the trajectory as evidence.
        """
        unavailable = {"critic_L2"}
        if not harness_status()[0]:
            unavailable.add("verify")
        return unavailable

    def _save_prediction(self, instance: dict, candidate: Candidate, run_id: str) -> Path:
        """Write the prediction in SWE-bench's expected format."""
        work_dir = (self.output_dir / "swe_harness").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        pred_path = work_dir / f"{run_id}.jsonl"
        pred_path.write_text(
            json.dumps(
                {
                    "instance_id": self.instance_id(instance),
                    "model_name_or_path": run_id,
                    "model_patch": candidate.payload or "",
                }
            )
            + "\n"
        )
        return pred_path

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from code_uq.scripts import spot_check_generators as scg

        usable, reason = harness_status()
        pred_path = self._save_prediction(instance, candidate, run_id)
        if not usable:
            # The patch is on disk either way, so labels can be computed later
            # on a host that has the harness; this episode simply has no label.
            return VerifyResult(
                False, detail=f"harness_unavailable: {reason}", available=False
            )

        work_dir = pred_path.parent
        try:
            report_path = scg.run_swebench_eval(
                predictions_path=pred_path,
                run_id=run_id,
                max_workers=self.harness_workers,
                work_dir=work_dir,
                dataset_name=self.dataset_name,
            )
            resolved = scg.parse_resolved(scg.load_report(report_path))
            ok = self.instance_id(instance) in resolved
            return VerifyResult(ok, detail=str(report_path))
        except Exception as exc:
            # An infrastructure error is not a verdict on the patch.
            log.warning("SWE harness failed for %s: %s", run_id, exc)
            return VerifyResult(
                False,
                detail=f"harness_error: {type(exc).__name__}: {exc}",
                available=False,
            )


def make_swe_adapter(
    benchmark: str,
    n_instances: int,
    seed: int,
    output_dir: Path,
    harness_workers: int,
) -> SWEAdapter:
    if benchmark == "swebench_lite":
        dataset = "princeton-nlp/SWE-bench_Lite"
    elif benchmark == "swebench_verified":
        dataset = "princeton-nlp/SWE-bench_Verified"
    else:
        raise ValueError(f"not a SWE benchmark: {benchmark}")
    return SWEAdapter(
        benchmark=benchmark,
        dataset_name=dataset,
        n_instances=n_instances,
        seed=seed,
        output_dir=output_dir,
        harness_workers=harness_workers,
    )
