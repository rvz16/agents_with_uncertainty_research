"""Every flag the cluster wrapper passes must exist in both runners.

A flag added to `run_in_container.sh` and to `run_alfworld.py` but not to
`run_alfworld_sharded.py` costs a full scheduling round trip to discover: the
task queues, installs vLLM, loads the model, and only then dies on
`unrecognized arguments`. That happened with the judge-tool flags after 33
minutes of GPU time.
"""
import re
from pathlib import Path

from experiments import run_alfworld, run_alfworld_sharded

WRAPPER = Path(__file__).resolve().parents[1] / "clearml" / "run_in_container.sh"


def _wrapper_flags() -> set[str]:
    """Only the arguments shared by both runners: `common_args` and its appends.

    `--workers` is deliberately outside that array -- it belongs to the sharded
    runner alone -- so reading the whole file would flag it wrongly.
    """
    text = WRAPPER.read_text()
    blocks = re.findall(r"common_args\+?=\((.*?)\n\s*\)", text, re.DOTALL)
    blocks += re.findall(r"common_args\+=\(([^()\n]*)\)", text)
    flags: set[str] = set()
    for block in blocks:
        flags.update(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", block))
    return flags


def _parser_flags(parser) -> set[str]:
    flags = set()
    for action in parser._actions:
        flags.update(action.option_strings)
    return flags


def test_both_runners_accept_every_flag_the_wrapper_sends():
    wrapper = _wrapper_flags()
    assert wrapper, "the wrapper parse found nothing, so this test proves nothing"

    single = _parser_flags(run_alfworld.build_parser())
    sharded = _parser_flags(run_alfworld_sharded.build_parser())

    for flag in sorted(wrapper):
        assert flag in single, f"{flag} is sent by the wrapper but run_alfworld rejects it"
        assert flag in sharded, (
            f"{flag} reaches run_alfworld but not run_alfworld_sharded, so any run "
            "with --workers > 1 dies after the model has loaded"
        )
