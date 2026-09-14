"""Re-score the steps whose answer was translated from a harmony tool call.

The run that introduced the translation (`_code_from_tool_call`) scored the
rebuilt text against the token stream, found no match, and left `combined`
empty on every translated step -- 2186 of 3543. The tokens are all stored,
so the fix (`split_at_last_message`) is applied to the recorded trajectories
instead of re-running the agent: the policy, the environment and the
outcomes are untouched; only the per-step metric bundles change.

    python -m experiments.rescore_translated RUN_DIR OUT_DIR
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from agents.react_agent import _certainties, _entropies, _metric_bundle, split_at_last_message
from uq.verbalized import parse_verbalized_confidence


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "run_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    rescored = total = 0
    with open(src / "trajectories.jsonl") as fin, open(dst / "trajectories.jsonl", "w") as fout:
        for line in fin:
            step = json.loads(line); total += 1
            if step.get("tool_call_translated") and step.get("token_logprobs"):
                reasoning, content = split_at_last_message(step["token_logprobs"])
                bundle = _metric_bundle([float(r["logprob"]) for r in content], _entropies(content), _certainties(content))
                bundle["verbalized_confidence"] = parse_verbalized_confidence(step.get("raw_response") or "")
                uq = step.setdefault("uq", {})
                uq["action"] = dict(bundle); uq["combined"] = dict(bundle)
                uq["reasoning"] = _metric_bundle([float(r["logprob"]) for r in reasoning])
                uq["reasoning"]["verbalized_confidence"] = bundle["verbalized_confidence"]
                step["logprobs_available"] = bool(content)
                step["perplexity"] = bundle["perplexity"]
                step["seqprob"] = bundle["sequence_probability"]
                rescored += 1
            fout.write(json.dumps(step) + "\n")
    print(f"rescored {rescored} of {total} steps -> {dst}")


if __name__ == "__main__":
    main()
