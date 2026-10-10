"""Fine-tune a small encoder to predict the outcome from a trajectory prefix.

The question is how much of a decision model's advantage is the trained
decision head and how much is simply a transformer reading the trajectory. So
this trains the smallest honest thing: ModernBERT over the same prefixes the
decision judge was given, with the episode's outcome as the label.

Prefixes, not episodes, are the examples: 442 ALFWorld episodes are too few to
fine-tune anything, while their prefixes number 6616. Folds are assigned by
episode, never by prefix, or an episode's own later steps would train the model
that scores its earlier ones.

Two outputs: the per-prefix probability, which is a per-step signal the report
can fuse like any other, and the episode-level answer file the scorer reads.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.prefix_judge import prompt

SEEDS = (0, 1, 2)
FOLDS = 5


def load(view_dir: Path, states_dir: Path, window: int, obs_chars: int, act_chars: int) -> dict[str, list]:
    """cohort -> [{id, label, score, prefixes: [str]}]."""
    labels: dict[tuple[str, str], tuple[int, float]] = {}
    for path in sorted(states_dir.glob("*.jsonl")):
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                score = float(row["score"])
                label = int(row["label"]) if row.get("label") is not None else None
                labels[(row["cohort"], row["id"])] = (label, score)
    out: dict[str, list] = {}
    for path in sorted(view_dir.glob("*.jsonl")):
        cohort = path.stem
        episodes = []
        for line in open(path):
            if not line.strip():
                continue
            view = json.loads(line)
            known = labels.get((cohort, view["id"]))
            if known is None:
                continue
            label, score = known
            prefixes = [prompt(view["task"], view["steps"][: k + 1], window, obs_chars, act_chars)
                        for k in range(len(view["steps"]))]
            if not prefixes:
                continue
            episodes.append({"id": view["id"], "label": label, "score": score, "prefixes": prefixes})
        if episodes:
            out[cohort] = episodes
    return out


def folds_of(episodes: list[dict], seed: int) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(episodes):
        by_label[int(e["label"])].append(i)
    folds: list[list[int]] = [[] for _ in range(FOLDS)]
    for label, members in sorted(by_label.items()):
        order = rng.permutation(members)
        for position, index in enumerate(order):
            folds[position % FOLDS].append(int(index))
    return folds


def train_once(model_id: str, train_rows, train_labels, eval_rows, args) -> list[float]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1).to(args.device)
    model.gradient_checkpointing_disable()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    order = np.random.RandomState(0).permutation(len(train_rows))
    batches = [order[i: i + args.batch] for i in range(0, len(order), args.batch)]
    steps = max(1, len(batches) * args.epochs)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps,
                                                   pct_start=0.1, anneal_strategy="linear")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(args.epochs):
        for batch in batches:
            texts = [train_rows[i] for i in batch]
            y = torch.tensor([train_labels[i] for i in batch], dtype=torch.float32, device=args.device)
            encoded = tokenizer(texts, truncation=True, max_length=args.max_length,
                                padding=True, return_tensors="pt").to(args.device)
            logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)

    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(eval_rows), args.eval_batch):
            texts = eval_rows[i: i + args.eval_batch]
            encoded = tokenizer(texts, truncation=True, max_length=args.max_length,
                                padding=True, return_tensors="pt").to(args.device)
            out += torch.sigmoid(model(**encoded).logits.squeeze(-1)).float().cpu().tolist()
    del model
    torch.cuda.empty_cache()
    return out


def flatten(episodes: list[dict], indices) -> tuple[list[str], list[float], list[tuple[int, int]]]:
    rows, labels, owner = [], [], []
    for i in indices:
        e = episodes[i]
        for k, text in enumerate(e["prefixes"]):
            rows.append(text); labels.append(float(e["label"])); owner.append((i, k))
    return rows, labels, owner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--states", type=Path, default=Path("data/decision_states_clean"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--act-chars", type=int, default=300)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--ood", action="store_true", help="also train on each cohort and score the others")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave a cohort or direction whose output file is already there; the output\n                        directory is mounted from the host, so a rerun need not retrain everything")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    data = load(a.views, a.states, a.window, a.obs_chars, a.act_chars)
    for cohort, episodes in data.items():
        print(f"[finetune] {cohort}: {len(episodes)} episodes, "
              f"{sum(len(e['prefixes']) for e in episodes)} prefixes", flush=True)

    for cohort, episodes in data.items():
        if a.skip_existing and (a.out / f"answers_modernbert_last_{cohort}.jsonl").exists():
            print(f"[finetune] {cohort} already written, skipping", flush=True)
            continue
        per_seed: list[dict[tuple[int, int], float]] = []
        for seed in SEEDS:
            predictions: dict[tuple[int, int], float] = {}
            for fold in folds_of(episodes, seed):
                held = set(fold)
                train_idx = [i for i in range(len(episodes)) if i not in held]
                train_rows, train_labels, _ = flatten(episodes, train_idx)
                eval_rows, _, owner = flatten(episodes, fold)
                values = train_once(a.model, train_rows, train_labels, eval_rows, a)
                for key, value in zip(owner, values):
                    predictions[key] = value
            per_seed.append(predictions)
            print(f"[finetune] {cohort} seed {seed} done", flush=True)
        keys = sorted(per_seed[0])
        mean = {k: st.fmean(s[k] for s in per_seed) for k in keys}

        with open(a.out / f"steps_{cohort}.jsonl", "w") as handle:
            for (i, k), value in sorted(mean.items()):
                handle.write(json.dumps({"id": episodes[i]["id"], "step": k, "p": value}) + "\n")
        for aggregate in ("last", "mean"):
            with open(a.out / f"answers_modernbert_{aggregate}_{cohort}.jsonl", "w") as handle:
                for i, e in enumerate(episodes):
                    values = [mean[(i, k)] for k in range(len(e["prefixes"])) if (i, k) in mean]
                    if not values:
                        continue
                    value = values[-1] if aggregate == "last" else st.fmean(values)
                    handle.write(json.dumps({"id": e["id"], "cohort": cohort, "label": e["label"],
                                             "score": e["score"],
                                             "answer": {"model": a.model,
                                                        "answers": {"success": {"type": "noul", "noul": value}}}}) + "\n")
        print(f"[finetune] {cohort} written", flush=True)

    if a.ood:
        for source, src_eps in data.items():
            rows, labels, _ = flatten(src_eps, range(len(src_eps)))
            for target, tgt_eps in data.items():
                if target == source:
                    continue
                if a.skip_existing and (a.out / f"ood_{source}_to_{target}.jsonl").exists():
                    print(f"[finetune] {source} -> {target} already written, skipping", flush=True)
                    continue
                eval_rows, _, owner = flatten(tgt_eps, range(len(tgt_eps)))
                values = train_once(a.model, rows, labels, eval_rows, a)
                by_episode: dict[int, list[float]] = defaultdict(list)
                for (i, _), value in zip(owner, values):
                    by_episode[i].append(value)
                with open(a.out / f"ood_{source}_to_{target}.jsonl", "w") as handle:
                    for i, e in enumerate(tgt_eps):
                        values_i = by_episode.get(i) or []
                        if not values_i:
                            continue
                        handle.write(json.dumps({"id": e["id"], "cohort": target, "label": e["label"],
                                                 "score": e["score"],
                                                 "answer": {"model": a.model,
                                                            "answers": {"success": {"type": "noul",
                                                                                    "noul": values_i[-1]}}}}) + "\n")
                print(f"[finetune] {source} -> {target} written", flush=True)
    print("[finetune] done")


if __name__ == "__main__":
    main()
