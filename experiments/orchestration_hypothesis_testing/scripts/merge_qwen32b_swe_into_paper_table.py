"""Move qwen25_32b entries from swebench_*_qwen32b cells into the main
swebench_lite / swebench_verified keys, so fig1_headline shows the full 5×5
SWE matrix."""
import json
from pathlib import Path

p = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/data/PAPER_TABLE.json")
d = json.loads(p.read_text())

merges = [("swebench_lite_qwen32b", "swebench_lite"),
          ("swebench_verified_qwen32b", "swebench_verified")]

for src_key, dst_key in merges:
    if src_key not in d:
        print(f"  no {src_key} key, skip")
        continue
    if dst_key not in d:
        d[dst_key] = {}
    for gen, variants in d[src_key].items():
        d[dst_key][gen] = variants
        print(f"  merged {src_key}/{gen} -> {dst_key}/{gen} ({len(variants)} variants)")
    del d[src_key]
    print(f"  removed top-level {src_key}")

p.write_text(json.dumps(d, indent=2))
print(f"\nfinal cells: {[k for k in d]}")
print(f"swebench_lite gens: {list(d['swebench_lite'].keys())}")
print(f"swebench_verified gens: {list(d['swebench_verified'].keys())}")
