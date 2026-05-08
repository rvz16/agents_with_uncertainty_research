"""Inject qwen25_32b SWE-Lite/Verified policy results into PAPER_TABLE.json
under swebench_lite.qwen25_32b and swebench_verified.qwen25_32b.

Reads:
  data/swebench_{lite,verified}_qwen32b/qwen25_32b/critic_results.jsonl
  data/swebench_{lite,verified}_qwen32b/qwen25_32b/likelihood_tables.json

Computes Bayesian/threshold/etc policies per instance, paired-bootstrap CI
on the diff vs always_verify, writes to PAPER_TABLE.json schema.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

EXP = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing")

# Cost vector matched to slide 51 / slide 53 (4x4 grid panel R=100, c_ver=1)
# so the qwen32b SWE entries are computed at the SAME operating point as the
# closed-API SWE cells already in PAPER_TABLE.
COST_GEN, COST_L0, COST_L3, COST_VER, REWARD = 5, 0.05, 0.05, 1, 100


def policy_always_verify(rec):
    return REWARD * rec["Y"] - COST_GEN - COST_VER

def policy_threshold_L0(rec):
    if rec.get("L0_syntax"):
        return REWARD * rec["Y"] - COST_GEN - COST_L0 - COST_VER
    return -COST_GEN - COST_L0

def policy_threshold_L3(rec):
    if rec.get("L3_llm_review"):
        return REWARD * rec["Y"] - COST_GEN - COST_L3 - COST_VER
    return -COST_GEN - COST_L3

def policy_fixed_pipeline(rec):
    if rec.get("L0_syntax") and rec.get("L3_llm_review"):
        return REWARD * rec["Y"] - COST_GEN - COST_L0 - COST_L3 - COST_VER
    return -COST_GEN - COST_L0 - COST_L3

def policy_bayesian_greedy(rec, prior, l3_y1, l3_y0):
    l3 = rec.get("L3_llm_review", False)
    if l3:
        post = (prior * l3_y1) / (prior * l3_y1 + (1 - prior) * l3_y0 + 1e-9)
    else:
        post = (prior * (1 - l3_y1)) / (prior * (1 - l3_y1) + (1 - prior) * (1 - l3_y0) + 1e-9)
    if post > 0.5:
        return REWARD * rec["Y"] - COST_GEN - COST_L3 - COST_VER
    return -COST_GEN - COST_L3

def policy_best_of_3(records_for_inst):
    """Take last patch (idx 2), verify."""
    last = records_for_inst[-1]
    return REWARD * last["Y"] - 3 * COST_GEN - COST_VER


def evaluate_cell(critic_path, lik_path):
    recs = [json.loads(l) for l in open(critic_path)]
    recs = [r for r in recs if r.get("Y") is not None]
    lik = json.loads(lik_path.read_text())
    prior = lik["prior_Y1"]
    l3_y1 = lik["critic_likelihoods"]["L3_llm_review"]["P_pass_given_Y1"]
    l3_y0 = lik["critic_likelihoods"]["L3_llm_review"]["P_pass_given_Y0"]

    by_inst = defaultdict(list)
    for r in recs:
        by_inst[r["instance_id"]].append(r)
    for v in by_inst.values():
        v.sort(key=lambda x: x.get("patch_id", 0))

    # per-instance policy utility (use first patch for stateless policies)
    inst_utils = {}
    for inst, lst in by_inst.items():
        first = lst[0]
        inst_utils[inst] = {
            "always_verify": policy_always_verify(first),
            "threshold_L0": policy_threshold_L0(first),
            "threshold_L3": policy_threshold_L3(first),
            "fixed_pipeline": policy_fixed_pipeline(first),
            "bayesian_greedy": policy_bayesian_greedy(first, prior, l3_y1, l3_y0),
            "bayesian_DP": policy_bayesian_greedy(first, prior, l3_y1, l3_y0),  # equiv for 1-step
            "best_of_3": policy_best_of_3(lst),
            "Y": first["Y"],
        }

    instances = list(inst_utils.keys())
    n = len(instances)
    pol_names = ["always_verify", "threshold_L0", "threshold_L3", "fixed_pipeline",
                 "bayesian_greedy", "bayesian_DP", "best_of_3"]

    # paired bootstrap on (policy - always_verify) per instance
    B = 1000
    means = {p: sum(inst_utils[i][p] for i in instances) / n for p in pol_names}
    pass_rates = {p: sum(1 for i in instances if inst_utils[i][p] > 0) / n for p in pol_names}
    diff = {p: means[p] - means["always_verify"] for p in pol_names}

    ci = {p: (0, 0) for p in pol_names}
    for p in pol_names:
        if p == "always_verify":
            continue
        diffs = [inst_utils[i][p] - inst_utils[i]["always_verify"] for i in instances]
        bs = []
        for _ in range(B):
            sample = [diffs[random.randrange(n)] for _ in range(n)]
            bs.append(sum(sample) / n)
        bs.sort()
        ci[p] = (bs[int(0.025 * B)], bs[int(0.975 * B)])

    policies = {}
    for p in pol_names:
        policies[p] = {
            "mean_utility": means[p],
            "pass_rate": pass_rates[p],
            "diff_vs_always_verify": diff[p],
            "ci95_lo": ci[p][0],
            "ci95_hi": ci[p][1],
        }

    return {
        "prior_Y1": prior,
        "L0_gap": lik["critic_likelihoods"]["L0_syntax"]["gap"],
        "L2_gap": lik["critic_likelihoods"].get("L2_public_tests", {}).get("gap", 0.0),
        "L3_gap_used": lik["critic_likelihoods"]["L3_llm_review"]["gap"],
        "policies": policies,
    }


def main():
    paper_path = EXP / "data/PAPER_TABLE.json"
    table = json.loads(paper_path.read_text())

    for bench in ["lite", "verified"]:
        cell = f"swe_{bench}"
        critic = EXP / f"data/swebench_{bench}_qwen32b/qwen25_32b/critic_results.jsonl"
        lik = EXP / f"data/swebench_{bench}_qwen32b/qwen25_32b/likelihood_tables.json"
        if not critic.exists() or not lik.exists():
            print(f"  {cell}: missing data, skip")
            continue
        result = evaluate_cell(critic, lik)
        if cell not in table:
            table[cell] = {}
        table[cell]["qwen25_32b"] = {"haiku45_default": result}
        print(f"  {cell}/qwen25_32b: prior={result['prior_Y1']:.3f}, "
              f"BG Δ={result['policies']['bayesian_greedy']['diff_vs_always_verify']:+.2f}, "
              f"CI=[{result['policies']['bayesian_greedy']['ci95_lo']:+.2f}, "
              f"{result['policies']['bayesian_greedy']['ci95_hi']:+.2f}]")

    paper_path.write_text(json.dumps(table, indent=2))
    print(f"\nfinal swebench_lite gens: {list(table['swe_lite'].keys())}")
    print(f"final swebench_verified gens: {list(table['swe_verified'].keys())}")


if __name__ == "__main__":
    main()
