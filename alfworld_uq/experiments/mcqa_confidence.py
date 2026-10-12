"""Does confidence work at all, on a task where nothing is hard?

Everything else here measures confidence about a trajectory: a long, partly
observed process whose outcome depends on dozens of decisions. If the signals
look weak there, the obvious question is whether the problem is trajectories or
whether confidence simply does not work for these models. This answers that by
taking the easiest possible case -- a single multiple-choice question, one
forward pass, the answer either right or wrong -- and measuring the same things
in the same way.

Three readings per question, all from one pass. ``option`` renormalises the
probability over the option letters, which is the analogue of a noul. ``margin``
is the gap between the best and second-best option, a confidence that ignores
the absolute scale. ``verbal`` asks the model for an integer and generates it,
the way our prefix judge does.

The metrics are the ones used throughout: accuracy, expected calibration error,
PRR of the confidence against correctness, and a selective-prediction curve,
since abstention is what a confidence is for -- refusing the questions it is
least sure of should raise accuracy on the rest, and by how much is the number
that matters.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

LETTERS = "ABCD"
TEMPLATE = ("Answer the multiple-choice question with a single letter.\n\n"
            "{question}\n{options}\nAnswer:")
VERBAL = ("Answer the multiple-choice question, then state your confidence.\n\n"
          "{question}\n{options}\n"
          "Reply with the letter, then a space, then an integer from 0 to 100: "
          "the probability that your answer is correct.\nAnswer:")


def ece(correct: list[int], confidence: list[float], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        pair = [(c, p) for c, p in zip(correct, confidence) if (p > low or b == 0) and p <= high]
        if pair:
            total += len(pair) / len(correct) * abs(
                st.fmean(c for c, _ in pair) - st.fmean(p for _, p in pair))
    return total


def _order(confidence: list[float], descending: bool) -> list[int]:
    """Ties broken at random, not by index: a constant confidence must score as
    uninformative rather than as whatever the input order happens to be."""
    import random
    rng = random.Random(0)
    keys = [(c, rng.random()) for c in confidence]
    return sorted(range(len(confidence)), key=lambda i: keys[i], reverse=descending)


def selective(correct: list[int], confidence: list[float]) -> list[tuple[float, float]]:
    """Accuracy on the kept share, answering the most confident questions first."""
    order = _order(confidence, True)
    out = []
    for share in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
        keep = order[: max(1, int(round(share * len(order))))]
        out.append((share, st.fmean(correct[i] for i in keep)))
    return out


def prr(correct: list[int], confidence: list[float]) -> float:
    """The same rejection-area ratio the trajectory tables use."""
    def area(key: list[float]) -> float:
        order = _order(key, False)
        kept, total = [], 0.0
        for cut in range(len(order)):
            rest = order[cut:]
            total += st.fmean(correct[i] for i in rest)
            kept.append(rest)
        return total / len(order)
    base = st.fmean(correct)
    a, o = area(confidence), area([float(c) for c in correct])
    return (a - base) / (o - base) if abs(o - base) > 1e-9 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dataset", default="cais/mmlu")
    p.add_argument("--subset", default="all")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["option", "verbal", "both"], default="both")
    p.add_argument("--max-new-tokens", type=int, default=8)
    a = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(a.dataset, a.subset, split=a.split)
    if a.limit and len(data) > a.limit:
        data = data.shuffle(seed=0).select(range(a.limit))
    print(f"[mcqa] {len(data)} questions from {a.dataset}/{a.subset}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    letter_ids = []
    for letter in LETTERS:
        ids = [tokenizer.encode(form, add_special_tokens=False) for form in (letter, " " + letter)]
        letter_ids.append([i[0] for i in ids if len(i) == 1])

    rows = []
    for index, item in enumerate(data, 1):
        choices = list(item["choices"])[:4]
        if len(choices) < 2:
            continue
        options = "\n".join(f"{LETTERS[k]}. {c}" for k, c in enumerate(choices))
        prompt = TEMPLATE.format(question=item["question"].strip(), options=options)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        batch = tokenizer(text, return_tensors="pt").to(a.device)
        with torch.no_grad():
            logits = model(**batch).logits[0, -1].float()
        probabilities = torch.softmax(logits, dim=-1)
        shares = [float(sum(probabilities[i] for i in ids)) if ids else 0.0 for ids in letter_ids]
        shares = shares[: len(choices)]
        total = sum(shares)
        shares = [s / total for s in shares] if total > 0 else [1 / len(choices)] * len(choices)
        best = max(range(len(shares)), key=lambda k: shares[k])
        ordered = sorted(shares, reverse=True)
        row = {"id": index, "answer": int(item["answer"]), "predicted": best,
               "correct": int(best == int(item["answer"])),
               "option": shares[best],
               "margin": ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)}

        if a.mode in ("verbal", "both"):
            vtext = tokenizer.apply_chat_template(
                [{"role": "user", "content": VERBAL.format(question=item["question"].strip(),
                                                           options=options)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            vbatch = tokenizer(vtext, return_tensors="pt").to(a.device)
            with torch.no_grad():
                generated = model.generate(**vbatch, max_new_tokens=a.max_new_tokens, do_sample=False)
            out_text = tokenizer.decode(generated[0, vbatch["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            numbers = [int(n) for n in __import__("re").findall(r"\d{1,3}", out_text)]
            row["verbal"] = min(max(numbers[-1], 0), 100) / 100 if numbers else None
            row["verbal_text"] = out_text[:60]
        rows.append(row)
        if index % 200 == 0:
            print(f"[mcqa] {index}/{len(data)} acc {st.fmean(r['correct'] for r in rows):.3f}",
                  flush=True)

    name = a.model.replace("/", "_")
    path = a.out / f"mcqa_{name}_{a.subset}.jsonl"
    with open(path, "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)

    correct = [r["correct"] for r in rows]
    print(f"\n=== {a.model} on {a.dataset}/{a.subset}: {len(rows)} questions, "
          f"accuracy {st.fmean(correct):.3f} ===")
    print(f"{'signal':10} {'ECE':>7} {'PRR':>7}   selective accuracy at coverage")
    for signal in ("option", "margin", "verbal"):
        values = [r.get(signal) for r in rows]
        pair = [(c, v) for c, v in zip(correct, values) if v is not None]
        if len(pair) < 20:
            continue
        c2, v2 = [c for c, _ in pair], [v for _, v in pair]
        curve = "  ".join(f"{int(s*100)}%:{acc:.3f}" for s, acc in selective(c2, v2))
        print(f"{signal:10} {ece(c2, v2):7.3f} {prr(c2, v2):7.3f}   {curve}")
    print(f"\n[mcqa] wrote {path}")


if __name__ == "__main__":
    main()
