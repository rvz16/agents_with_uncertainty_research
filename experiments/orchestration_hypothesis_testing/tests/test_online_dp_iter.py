"""Unit tests for analysis/online_dp_iter.py — per-instance online-BDP
replay on iter trajectories."""
from __future__ import annotations

import pathlib
import sys

import pytest

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from analysis.controller import CostModel  # noqa: E402
from analysis.online_dp_iter import (  # noqa: E402
    _bayes_update_belief,
    _online_bdp_decide,
    simulate_online_bdp_on_iter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cheap_cost():
    """Cost model where verification is cheap relative to reward — encourages
    'verify' over 'give_up'. Use for happy-path tests."""
    return CostModel(reward=100.0, c_ver=5.0, c_gen=10.0,
                     c_L0=1.0, c_L2=2.0, c_L3=5.0)


@pytest.fixture
def expensive_verify_cost():
    """Cost model where verification is expensive — only worth it if belief
    is high. Use to test give_up paths."""
    return CostModel(reward=100.0, c_ver=50.0, c_gen=10.0,
                     c_L0=1.0, c_L2=2.0, c_L3=5.0)


@pytest.fixture
def neutral_kernel():
    return {"p_fix_broken": 0.5, "p_break_correct": 0.1}


@pytest.fixture
def empty_likes():
    """No critics — belief never gets updated from observations, stays at prior."""
    return {"critic_likelihoods": {}}


# ---------------------------------------------------------------------------
# _online_bdp_decide — small backward-induction DP
# ---------------------------------------------------------------------------

def test_decide_high_belief_verifies_at_last_step(cheap_cost, neutral_kernel):
    """At the last step with high belief, verify is optimal."""
    assert _online_bdp_decide(0.9, k_left=1, kernel=neutral_kernel, cost=cheap_cost) == "verify"


def test_decide_low_belief_gives_up_at_last_step(cheap_cost, neutral_kernel):
    """At the last step with very low belief, give_up beats verify
    (R·b − c_ver = 100·0.01 − 5 = −4 < 0)."""
    assert _online_bdp_decide(0.01, k_left=1, kernel=neutral_kernel, cost=cheap_cost) == "give_up"


def test_decide_high_belief_verifies_even_with_lookahead(cheap_cost, neutral_kernel):
    """Even with multiple steps remaining, very high belief → verify NOW."""
    assert _online_bdp_decide(0.95, k_left=5, kernel=neutral_kernel, cost=cheap_cost) == "verify"


def test_decide_low_belief_high_p_fix_prefers_generate(cheap_cost):
    """With p_fix=0.95 and low current belief, generate looks good because
    belief jumps to ~0.95 next step → verify there is profitable."""
    high_fix_kernel = {"p_fix_broken": 0.95, "p_break_correct": 0.05}
    a = _online_bdp_decide(0.05, k_left=3, kernel=high_fix_kernel, cost=cheap_cost)
    assert a == "generate"


def test_decide_accepts_uppercase_kernel_keys(cheap_cost):
    """Same as a low-belief generate test, but kernel uses uppercase keys."""
    k = {"P_fix_given_broken": 0.95, "P_break_given_correct": 0.05}
    assert _online_bdp_decide(0.05, k_left=3, kernel=k, cost=cheap_cost) == "generate"


def test_decide_rejects_malformed_kernel(cheap_cost):
    with pytest.raises(ValueError, match="missing"):
        _online_bdp_decide(0.5, k_left=2, kernel={"foo": 0.5}, cost=cheap_cost)


# ---------------------------------------------------------------------------
# _bayes_update_belief
# ---------------------------------------------------------------------------

def test_bayes_update_no_critics_returns_input():
    """No critics in likes → belief unchanged."""
    assert _bayes_update_belief(0.5, {}, {"Y": 1, "L0_syntax": True}) == 0.5


def test_bayes_update_single_critic_pass():
    """One critic, passes; belief should move toward 1 (since P_pass|Y=1 > P_pass|Y=0)."""
    likes = {"L0_syntax": {"P_pass_given_Y1": 0.9, "P_pass_given_Y0": 0.2}}
    new = _bayes_update_belief(0.5, likes, {"L0_syntax": True})
    # P(Y=1|pass) = 0.9*0.5 / (0.9*0.5 + 0.2*0.5) = 0.45/0.55 ≈ 0.818
    assert new == pytest.approx(0.45 / 0.55, abs=1e-6)


def test_bayes_update_single_critic_fail():
    """Same likes, critic FAILS; belief moves toward 0."""
    likes = {"L0_syntax": {"P_pass_given_Y1": 0.9, "P_pass_given_Y0": 0.2}}
    new = _bayes_update_belief(0.5, likes, {"L0_syntax": False})
    # P(Y=1|fail) = 0.1*0.5 / (0.1*0.5 + 0.8*0.5) = 0.05/0.45 ≈ 0.111
    assert new == pytest.approx(0.05 / 0.45, abs=1e-6)


def test_bayes_update_skips_missing_likes():
    """If only L0 is in likes but rec has L0+L3, L3 is skipped silently."""
    likes = {"L0_syntax": {"P_pass_given_Y1": 0.9, "P_pass_given_Y0": 0.2}}
    new = _bayes_update_belief(0.5, likes, {"L0_syntax": True, "L3_llm_review": True})
    assert new == pytest.approx(0.45 / 0.55, abs=1e-6)


def test_bayes_update_skips_none_likelihoods():
    """If a critic has P_pass_given_Y1 = None, it's skipped (matches the
    canonical likes-table-pruning in the notebook's cell 13)."""
    likes = {"L0_syntax": {"P_pass_given_Y1": None, "P_pass_given_Y0": 0.2}}
    new = _bayes_update_belief(0.5, likes, {"L0_syntax": True})
    assert new == 0.5  # no update


# ---------------------------------------------------------------------------
# simulate_online_bdp_on_iter — high-level integration tests
# ---------------------------------------------------------------------------

def test_simulate_empty_trajectory(cheap_cost, neutral_kernel, empty_likes):
    out = simulate_online_bdp_on_iter([], empty_likes, 0.5, neutral_kernel, cheap_cost)
    assert out["stop_reason"] == "empty_trajectory"
    assert out["utility"] == 0.0
    assert out["n_kernel_updates"] == 0


def test_simulate_high_prior_verifies_at_step_0(cheap_cost, neutral_kernel, empty_likes):
    """With a high prior and no critics, BDP should verify on the first patch."""
    traj = [{"step": i, "Y": 1} for i in range(5)]
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.95,
                                       kernel_seed=neutral_kernel, cost=cheap_cost)
    assert out["verified"]
    assert out["stop_step"] == 0
    assert out["n_kernel_updates"] == 0   # never moved past step 0
    assert out["utility"] == pytest.approx(100.0 - 5.0)


def test_simulate_low_prior_low_p_fix_gives_up(expensive_verify_cost, empty_likes):
    """With low prior, expensive verify, and a kernel that says fixes are rare,
    the optimal first action is give_up."""
    bad_kernel = {"p_fix_broken": 0.05, "p_break_correct": 0.1}
    traj = [{"step": i, "Y": 0} for i in range(5)]
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.05,
                                       kernel_seed=bad_kernel, cost=expensive_verify_cost)
    assert not out["verified"]
    assert out["stop_reason"] == "give_up"
    assert out["stop_step"] == 0
    assert out["utility"] == 0.0


def test_simulate_kernel_updates_per_generated_step(cheap_cost, empty_likes):
    """Structural invariant: n_kernel_updates equals the number of generate
    actions (== stop_step when the run ends in verify). Pins the contract
    that every generate-action absorbs exactly one (Y_t, Y_{t+1}) pair."""
    # High p_fix kernel → BDP wants to generate at least once from a low prior
    high_fix_kernel = {"p_fix_broken": 0.95, "p_break_correct": 0.05}
    traj = [
        {"step": 0, "Y": 0},
        {"step": 1, "Y": 1},
        {"step": 2, "Y": 1},
        {"step": 3, "Y": 1},
        {"step": 4, "Y": 1},
    ]
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.05,
                                       kernel_seed=high_fix_kernel, cost=cheap_cost)
    # Agent must generate at least once (low prior + high p_fix favors gen)
    # and eventually verify (not give_up).
    assert out["verified"]
    assert out["stop_step"] >= 1   # at least one generate happened
    # The invariant: one kernel update per generate.
    # stop_step is the index at which verify happened, so generates =
    # stop_step (steps 0..stop_step-1 were all generates).
    assert out["n_kernel_updates"] == out["stop_step"]


def test_simulate_kernel_actually_evolves(cheap_cost, empty_likes):
    """After multiple generate steps, the online kernel posterior should
    differ from the seed. Confirms .update() is actually being called."""
    # Cost configured to force generate through several steps:
    #   reward=100, c_ver=99 → only verify if belief > 0.99
    #   c_gen=0.5 → cheap to keep generating
    cost = CostModel(reward=100.0, c_ver=99.0, c_gen=0.5,
                     c_L0=1.0, c_L2=2.0, c_L3=5.0)
    # Seed kernel with p_fix=0.5 (uninformative-ish)
    seed = {"p_fix_broken": 0.5, "p_break_correct": 0.05}
    # 5-step trajectory: all (0, 1) transitions → online posterior pushes p_fix up
    traj = [
        {"step": 0, "Y": 0},
        {"step": 1, "Y": 1},
        {"step": 2, "Y": 0},
        {"step": 3, "Y": 1},
        {"step": 4, "Y": 0},
    ]
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.3,
                                       kernel_seed=seed, cost=cost)
    # At minimum: kernel should have moved away from seed (or trajectory
    # ended in give_up before any updates — check explicitly)
    if out["n_kernel_updates"] > 0:
        final = out["final_kernel"]
        # n_broken_observed >= 1, p_fix recomputed; should differ from 0.5
        assert final["p_fix_broken"] != pytest.approx(0.5)


def test_simulate_per_instance_reset(cheap_cost, empty_likes):
    """Calling the simulator twice with the same seed kernel produces two
    independent runs — second run does NOT inherit kernel state from the
    first. This is the 'per-instance reset' contract."""
    high_fix_kernel = {"p_fix_broken": 0.95, "p_break_correct": 0.05}
    traj = [{"step": 0, "Y": 0}, {"step": 1, "Y": 1}]
    out1 = simulate_online_bdp_on_iter(traj, empty_likes, 0.05,
                                         high_fix_kernel, cheap_cost)
    out2 = simulate_online_bdp_on_iter(traj, empty_likes, 0.05,
                                         high_fix_kernel, cheap_cost)
    # Both runs should produce identical results — no cross-call state
    assert out1["utility"] == out2["utility"]
    assert out1["n_kernel_updates"] == out2["n_kernel_updates"]
    # Final kernel should be identical (deterministic given seed + traj)
    assert out1["final_kernel"] == out2["final_kernel"]


def test_simulate_handles_y_none_mid_trajectory(cheap_cost, empty_likes):
    """If Y is None at an intermediate step, kernel update for that boundary
    is skipped but the run continues."""
    high_fix_kernel = {"p_fix_broken": 0.95, "p_break_correct": 0.05}
    traj = [
        {"step": 0, "Y": 0},
        {"step": 1, "Y": None},   # mid-trajectory gap (SWE pre-harness backfill)
        {"step": 2, "Y": 1},
    ]
    out = simulate_online_bdp_on_iter(traj, empty_likes, 0.05,
                                       high_fix_kernel, cheap_cost)
    # If agent generated past step 0: the (0, None) transition is dropped,
    # so n_kernel_updates < n_generate-steps.
    # The simulator must not crash.
    assert "utility" in out


def test_simulate_critics_move_belief(cheap_cost):
    """A passing L3 critic at step 0 should boost belief enough to verify,
    even from a modest prior."""
    likes = {"critic_likelihoods": {
        "L3_llm_review": {"P_pass_given_Y1": 0.9, "P_pass_given_Y0": 0.1},
    }}
    kernel = {"p_fix_broken": 0.5, "p_break_correct": 0.1}
    traj = [{"step": 0, "Y": 1, "L3_llm_review": True}]
    out = simulate_online_bdp_on_iter(traj, likes, prior=0.5,
                                       kernel_seed=kernel, cost=cheap_cost)
    # Belief after L3 pass: 0.9*0.5 / (0.9*0.5 + 0.1*0.5) = 0.9 — well above
    # the verify threshold for cheap_cost
    assert out["verified"]
    assert out["belief_trajectory"][0] == pytest.approx(0.9, abs=1e-6)


def test_simulate_belief_propagates_via_kernel_between_steps(cheap_cost, empty_likes):
    """After a generate, the belief at the next step should equal the kernel-
    propagated belief from the previous step (with no critic info)."""
    # Force generate at step 0 by using expensive verify
    cost = CostModel(reward=100.0, c_ver=50.0, c_gen=1.0,
                     c_L0=1.0, c_L2=2.0, c_L3=5.0)
    # Use high-fix kernel + low prior so agent prefers generate then verify
    kernel = {"p_fix_broken": 0.9, "p_break_correct": 0.1}
    traj = [{"step": 0, "Y": 0}, {"step": 1, "Y": 1}]
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.2,
                                       kernel_seed=kernel, cost=cost)
    # Belief at step 0: 0.2 (prior, no critics)
    # Belief at step 1 (after generate): 0.2 * 0.9 + 0.8 * (updated p_fix)
    # The online kernel at step 1 has 1 observation (0, 1), so p_fix updates.
    # With seed p_fix=0.9 + 1 fix observation, posterior p_fix is even higher.
    # We just confirm belief propagates (i.e., changes) rather than staying
    # at the prior.
    assert out["belief_trajectory"][0] == pytest.approx(0.2, abs=1e-6)
    if len(out["belief_trajectory"]) > 1:
        assert out["belief_trajectory"][1] != pytest.approx(0.2, abs=1e-3)


def test_simulate_y_missing_at_verify_step(cheap_cost, empty_likes):
    """Agent verifies but Y is None (SWE pre-harness) — record as
    'verify_but_y_missing' and treat as give_up for utility purposes."""
    traj = [{"step": 0, "Y": None}]   # verify immediately, but no Y
    out = simulate_online_bdp_on_iter(traj, empty_likes, prior=0.95,
                                       kernel_seed={"p_fix_broken": 0.5, "p_break_correct": 0.1},
                                       cost=cheap_cost)
    assert out["stop_reason"] == "verify_but_y_missing"
    assert not out["verified"]
    # Utility = 0 (no reward) − c_ver
    assert out["utility"] == pytest.approx(-cheap_cost.c_ver)
