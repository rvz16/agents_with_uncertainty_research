"""Our own decision model: a LoRA over a small backbone with a schema head.

The measurements this is built on are specific. A logistic regression over a
frozen 0.8B backbone already reaches .82 in distribution, and the released
decision heads reach .15 to .55, so there is nothing left to win in
distribution and no reason to pay for scale. What nobody survives is a change
of cohort: .88 falls to between .32 and .47, ours and theirs alike. So this
targets transfer, and everything in it is chosen for that.

Three things supervise the head, and they can be switched off one at a time so
each one's contribution is measurable rather than assumed.

* The verified outcome, under a proper scoring rule -- log loss with a Brier
  term. The released models add a reinforcement stage for probability matching;
  our evidence says it buys nothing a proper scoring rule does not, and this
  makes that an ablation instead of a belief.
* Our own posterior, as a soft target. The tempered tool-critic belief is the
  best-calibrated estimator we measured, at .041 expected calibration error,
  but it reads harness-specific critic observations and so does not travel.
  Reproducing it from the trajectory text is the point: the calibration without
  the dependence.
* The structural facts ALFWorld labels mechanically -- how much of the prefix
  repeats itself, and how much of it rests on a stale check. These are the
  conditional dependencies an additive log-likelihood ratio over token
  statistics cannot express, and they are the answers whose fusion took a
  released model from .22 to .76.

One warning the data forces. On the smolagents Qwen cohort our posterior is a
calibrated constant -- .019 expected calibration error at zero discrimination --
so distilling it there teaches the base rate and nothing else. The per-cohort
report is there to make that visible rather than let it average away.

Folds are cut by episode and expanded to prefixes, never by prefix.
``--shuffle-labels`` permutes within cohort and must come back at zero.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.prefix_judge import prompt

COHORTS = ("RG", "SG", "RQ", "SQ")
FOLDS = 5


def load(views: Path, targets: Path, window: int, obs_chars: int, act_chars: int) -> list[dict]:
    """One row per prefix: its text, the outcome, our posterior, the structural facts."""
    rows: list[dict] = []
    for cohort in COHORTS:
        view_path, target_path = views / f"{cohort}.jsonl", targets / f"targets_{cohort}.jsonl"
        if not (view_path.exists() and target_path.exists()):
            print(f"[train] {cohort}: missing views or targets, skipped", flush=True)
            continue
        steps_of = {}
        for line in open(view_path):
            if line.strip():
                view = json.loads(line)
                steps_of[view["id"]] = view
        by_episode: dict[str, list[dict]] = defaultdict(list)
        for line in open(target_path):
            if line.strip():
                by_episode[json.loads(line)["id"]].append(json.loads(line))
        for episode_id, targets_of in by_episode.items():
            view = steps_of.get(episode_id)
            if not view:
                continue
            for t in sorted(targets_of, key=lambda r: r["step"]):
                k = t["step"]
                if k >= len(view["steps"]):
                    continue
                rows.append({
                    "cohort": cohort, "episode": episode_id, "step": k,
                    "text": prompt(view["task"], view["steps"][: k + 1], window, obs_chars, act_chars),
                    "label": float(t["label"]), "posterior": float(t["posterior"]),
                    "repeat": t.get("repeat_share"), "stale": t.get("stale_share"),
                })
    print(f"[train] {len(rows)} prefixes over "
          f"{len({(r['cohort'], r['episode']) for r in rows})} episodes", flush=True)
    return rows


def permute(rows: list[dict]) -> list[dict]:
    """The control: move each episode's label to another episode of similar length.

    Permuting episodes outright changes the prefix-weighted rate of positives --
    0.485 to 0.602 on this data -- because episodes differ in length, which makes
    the control's calibration incomparable with the run it controls. Permuting
    inside length terciles keeps the rate and still destroys the association
    between a trajectory and its outcome, which is the only thing a control has
    to do.
    """
    rng = np.random.RandomState(12345)
    lengths: dict[tuple[str, str], int] = defaultdict(int)
    labels: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r["cohort"], r["episode"])
        lengths[key] += 1
        labels[key] = r["label"]
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for cohort in COHORTS:
        keys = sorted((k for k in lengths if k[0] == cohort), key=lambda k: lengths[k])
        for third in range(3):
            block = keys[third * len(keys) // 3: (third + 1) * len(keys) // 3]
            shuffled = list(block); rng.shuffle(shuffled)
            mapping.update(dict(zip(block, shuffled)))
    for r in rows:
        r["label"] = labels[mapping[(r["cohort"], r["episode"])]]
    print("[train] labels permuted inside length terciles: this run is the control", flush=True)
    return rows


def folds_by_episode(rows: list[dict], seed: int) -> list[list[int]]:
    episodes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        episodes[(r["cohort"], r["episode"])].append(i)
    keys = sorted(episodes)
    strata = {k: int(rows[episodes[k][0]]["label"] > 0.5) for k in keys}
    rng = np.random.RandomState(seed)
    folds: list[list[int]] = [[] for _ in range(FOLDS)]
    for value in (0, 1):
        members = [k for k in keys if strata[k] == value]
        rng.shuffle(members)
        for position, key in enumerate(members):
            folds[position % FOLDS] += episodes[key]
    return folds


class Head:
    """success, repeats, stale -- the three facts we can supervise."""

    OUTPUTS = 3


def build(model_id: str, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=args.device)
    if args.lora_rank:
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        backbone.print_trainable_parameters()
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    hidden = backbone.config.hidden_size
    head = torch.nn.Sequential(
        torch.nn.Linear(hidden, args.hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, Head.OUTPUTS)).to(args.device).to(torch.float32)
    return tokenizer, backbone, head


def forward(backbone, head, tokenizer, texts: list[str], args):
    import torch
    encoded = tokenizer(texts, truncation=True, max_length=args.max_length, padding=True,
                        return_tensors="pt").to(args.device)
    states = backbone(**encoded).last_hidden_state
    index = encoded["attention_mask"].sum(1) - 1
    pooled = states[torch.arange(states.shape[0], device=states.device), index]
    return head(pooled.to(torch.float32))


def train_fold(rows: list[dict], train_idx, eval_idx, args) -> list[dict]:
    import torch

    tokenizer, backbone, head = build(args.model, args)
    parameters = [p for p in list(backbone.parameters()) + list(head.parameters())
                  if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    order = np.random.RandomState(0).permutation(list(train_idx))
    batches = [order[i: i + args.batch] for i in range(0, len(order), args.batch)]
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(1, len(batches) * args.epochs),
        pct_start=0.1, anneal_strategy="linear")
    bce = torch.nn.BCEWithLogitsLoss()

    backbone.train(); head.train()
    for epoch in range(args.epochs):
        for number, batch in enumerate(batches, 1):
            texts = [rows[i]["text"] for i in batch]
            logits = forward(backbone, head, tokenizer, texts, args)
            device = logits.device
            y = torch.tensor([rows[i]["label"] for i in batch], dtype=torch.float32, device=device)
            q = torch.tensor([rows[i]["posterior"] for i in batch], dtype=torch.float32, device=device)
            p = torch.sigmoid(logits[:, 0])
            loss = args.w_outcome * bce(logits[:, 0], y) + args.w_brier * ((p - y) ** 2).mean()
            if args.w_distill:
                loss = loss + args.w_distill * ((p - q) ** 2).mean()
            if args.w_aux:
                for position, field in ((1, "repeat"), (2, "stale")):
                    have = [(j, rows[i][field]) for j, i in enumerate(batch)
                            if rows[i].get(field) is not None]
                    if have:
                        idx = torch.tensor([j for j, _ in have], device=device)
                        target = torch.tensor([v for _, v in have], dtype=torch.float32, device=device)
                        loss = loss + args.w_aux * bce(logits[idx, position], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
            if number % 50 == 0:
                print(f"[train] epoch {epoch + 1} batch {number}/{len(batches)} "
                      f"loss {float(loss):.4f}", flush=True)

    backbone.eval(); head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(eval_idx), args.eval_batch):
            chunk = list(eval_idx)[i: i + args.eval_batch]
            logits = forward(backbone, head, tokenizer, [rows[j]["text"] for j in chunk], args)
            probabilities = torch.sigmoid(logits).float().cpu().tolist()
            for j, values in zip(chunk, probabilities):
                out.append({"index": j, "success": values[0], "stuck": values[1], "stale": values[2]})
    del backbone, head
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--targets", type=Path, default=Path("data/distill_targets"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", default="ours")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lora-rank", type=int, default=16, help="0 freezes the backbone entirely")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--act-chars", type=int, default=300)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--w-outcome", type=float, default=1.0)
    p.add_argument("--w-brier", type=float, default=1.0)
    p.add_argument("--w-distill", type=float, default=1.0,
                   help="0 ablates the distillation of our posterior")
    p.add_argument("--w-aux", type=float, default=0.3,
                   help="0 ablates the structural questions")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="the control: permute within cohort, which must score at zero")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = load(a.views, a.targets, a.window, a.obs_chars, a.act_chars)
    if not rows:
        raise SystemExit("no prefixes: check --views and --targets")
    if a.shuffle_labels:
        rows = permute(rows)

    predictions: dict[int, list[dict]] = defaultdict(list)
    for seed in a.seeds:
        for number, fold in enumerate(folds_by_episode(rows, seed), 1):
            held = set(fold)
            train_idx = [i for i in range(len(rows)) if i not in held]
            print(f"[train] seed {seed} fold {number}/{FOLDS}: "
                  f"{len(train_idx)} train, {len(fold)} eval", flush=True)
            for row in train_fold(rows, train_idx, fold, a):
                predictions[row["index"]].append(row)

    tag = f"{a.tag}_shuffled" if a.shuffle_labels else a.tag
    name = a.model.split("/")[-1]
    per_cohort: dict[str, list] = defaultdict(list)
    for index, values in predictions.items():
        row = rows[index]
        per_cohort[row["cohort"]].append((row, {k: st.fmean(v[k] for v in values)
                                                for k in ("success", "stuck", "stale")}))
    for cohort, items in per_cohort.items():
        with open(a.out / f"steps_{tag}_{name}_{cohort}.jsonl", "w") as handle:
            for row, value in sorted(items, key=lambda t: (t[0]["episode"], t[0]["step"])):
                handle.write(json.dumps({"id": row["episode"], "step": row["step"],
                                         "p": value["success"], "stuck": value["stuck"],
                                         "stale": value["stale"]}) + "\n")
        last: dict[str, tuple] = {}
        for row, value in items:
            if row["episode"] not in last or row["step"] > last[row["episode"]][0]["step"]:
                last[row["episode"]] = (row, value)
        with open(a.out / f"answers_{tag}_{name}_{cohort}.jsonl", "w") as handle:
            for episode, (row, value) in sorted(last.items()):
                handle.write(json.dumps({
                    "id": episode, "cohort": cohort, "label": int(row["label"]),
                    "score": row["label"],
                    "answer": {"model": f"{a.model}+lora", "answers": {
                        "success": {"type": "noul", "noul": value["success"]}}}}) + "\n")
        print(f"[train] {cohort}: {len(last)} episodes, {len(items)} prefixes written", flush=True)
    print("[train] done")


if __name__ == "__main__":
    main()
