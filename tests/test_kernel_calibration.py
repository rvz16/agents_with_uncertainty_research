"""The transition kernel measured by the train-split refinement chain.

Guards the two properties that make the chain worth its own pass: pairs are
consecutive within an instance (never across instances), and the chain keeps
going after a success so transitions out of the correct state are observable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_uq.common.kernel import compute_transition_kernel_from_pairs


def pairs_from_rows(rows):
    """Same extraction the agent does after the chains finish."""
    latest = {}
    for row in rows:
        if row.get("passed") is None:
            continue
        latest[(str(row["instance_id"]), int(row.get("patch_id", 0)))] = int(bool(row["passed"]))
    by_instance = {}
    for (inst, patch), passed in latest.items():
        by_instance.setdefault(inst, []).append((patch, passed))
    pairs = []
    for seq in by_instance.values():
        ordered = [y for _, y in sorted(seq)]
        pairs.extend(zip(ordered, ordered[1:]))
    return pairs


def _row(inst, patch, passed):
    return {"instance_id": inst, "patch_id": patch, "passed": passed}


def test_pairs_are_consecutive_within_instance():
    rows = [_row("a", 0, False), _row("a", 1, True), _row("a", 2, True),
            _row("b", 0, True), _row("b", 1, False)]
    assert pairs_from_rows(rows) == [(0, 1), (1, 1), (1, 0)]


def test_pairs_never_cross_instance_boundaries():
    rows = [_row("a", 0, False), _row("b", 0, True)]
    assert pairs_from_rows(rows) == []


def test_out_of_order_patches_are_sorted():
    rows = [_row("a", 2, True), _row("a", 0, False), _row("a", 1, False)]
    assert pairs_from_rows(rows) == [(0, 0), (0, 1)]


def test_unverifiable_patches_are_dropped_not_counted_as_failure():
    rows = [_row("a", 0, True), _row("a", 1, None), _row("a", 2, False)]
    # the unlabelled patch drops out; the surviving pair is 1 -> 0
    assert pairs_from_rows(rows) == [(1, 0)]


def test_break_transitions_are_observable_when_chain_continues():
    """The whole reason the chain does not stop at the first success."""
    rows = [_row("a", 0, True), _row("a", 1, False)]
    kernel = compute_transition_kernel_from_pairs(pairs_from_rows(rows))
    assert kernel["n_correct_observed"] == 1
    assert kernel["raw_counts"]["1->0"] == 1


def test_kernel_differs_from_the_uncalibrated_placeholder():
    """A run of mostly-failing refinements must not come out at p_fix = 0.50."""
    rows = []
    for i in range(20):
        rows += [_row(f"i{i}", 0, False), _row(f"i{i}", 1, False), _row(f"i{i}", 2, False)]
    rows += [_row("fixed", 0, False), _row("fixed", 1, True)]
    kernel = compute_transition_kernel_from_pairs(pairs_from_rows(rows))
    assert kernel["P_fix_given_broken"] < 0.15, kernel["P_fix_given_broken"]
    assert kernel["n_pairs"] == 41


def test_empty_pairs_fall_back_to_the_beta_prior_not_a_crash():
    kernel = compute_transition_kernel_from_pairs([])
    assert kernel["n_pairs"] == 0
    assert 0.0 < kernel["P_fix_given_broken"] < 1.0


def test_default_output_path_never_collides_with_the_results_file():
    """The naive `.with_suffix(".jsonl")` lands on the test results file.

    "<stem>.train_prior_calibration.jsonl" has ".train_prior_calibration" as its
    suffix, so replacing the suffix drops the marker entirely and yields
    "<stem>.jsonl" — the file the analyzer reads as test episodes. Calibration
    rows appended there would be scored as if they were held-out instances.
    """
    prior = Path("runs/x/lcb_hard__gen.train_prior_calibration.jsonl")
    results = Path("runs/x/lcb_hard__gen.jsonl")

    naive = prior.with_suffix("").with_name(
        prior.stem.replace("train_prior_calibration", "train_kernel_calibration")
    ).with_suffix(".jsonl")
    assert naive == results, "this is the trap the real derivation must avoid"

    derived = prior.with_name(
        prior.name.replace("train_prior_calibration", "train_kernel_calibration")
    )
    assert derived != results
    assert derived.name == "lcb_hard__gen.train_kernel_calibration.jsonl"


def test_a_redone_chain_does_not_double_count_its_transitions():
    """Resume redoes a whole chain, so the file can hold two rows per patch."""
    rows = [_row("a", 0, False), _row("a", 1, True),          # первая попытка
            _row("a", 0, False), _row("a", 1, False)]         # цепочка переделана
    # учитывается только последняя пара, а не обе
    assert pairs_from_rows(rows) == [(0, 0)]
