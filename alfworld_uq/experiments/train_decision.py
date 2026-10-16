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

COHORTS = ("RG", "SG", "RQ", "SQ", "DG", "DL", "DQ")
#: the coding cohorts are scored on partial credit, so their target is the score
#: itself rather than a binary; a cross-entropy takes a soft target unchanged
CONTINUOUS = ("DG", "DL", "DQ")
FOLDS = 5


def load(views: Path, targets: Path, window: int, obs_chars: int, act_chars: int,
         task_only: bool = False) -> list[dict]:
    """One row per prefix: its text, the outcome, our posterior, the structural facts.

    With ``task_only`` the transcript is dropped and one row per episode remains:
    the baseline that says how much of an early estimate is reading the task
    rather than watching the agent. Our model scores .42 having seen a tenth of
    an episode, which at sixteen steps is one or two actions -- far too little to
    see anything go wrong, so the honest question is what the task text alone is
    worth.
    """
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
                if task_only and k:
                    continue  # every prefix would carry the same text
                rows.append({
                    "cohort": cohort, "episode": episode_id, "step": k,
                    "n_steps": len(view["steps"]),
                    "text": prompt(view["task"], [] if task_only else view["steps"][: k + 1],
                                    window, obs_chars, act_chars),
                    "label": float(t["score"] if cohort in CONTINUOUS else t["label"]),
                    "posterior": float(t["posterior"]),
                    "repeat": t.get("repeat_share"), "stale": t.get("stale_share"),
                })
    print(f"[train] {len(rows)} prefixes over "
          f"{len({(r['cohort'], r['episode']) for r in rows})} episodes", flush=True)
    return rows


def utilities(row: dict, c_step: float, c_ask: float) -> tuple[float, float, float]:
    """What each action would really have been worth on this prefix.

    The episode was recorded to the end, so this is not an estimate: we know the
    counterfactual payoff of every action, and the accounting is the one
    gate_utility scores the threshold policies with.

      continue  the episode runs on and pays for every step it took
      ask       a human is paid for, and the task is then solved
      stop      nothing is gained, and the steps so far are still paid for

    Because all three are known, the controller does not need a policy gradient:
    the expected utility of a stochastic policy is an exact differentiable
    function of its probabilities.
    """
    k = row["step"]
    return (row["label"] - c_step * row["n_steps"],
            1.0 - c_step * (k + 1) - c_ask,
            -c_step * (k + 1))


ENVS = {"alfworld": ("RG", "SG", "RQ", "SQ"), "deepswe": ("DG", "DL", "DQ")}


def cohort_spec(spec: str) -> tuple[str, ...]:
    """Resolve "SQ", "deepswe", or "RQ,SQ" to cohort names.

    A single cohort is not enough for a cross-domain claim: holding out DG leaves
    DL and DQ in training, which is transfer between two DeepSWE cohorts, not
    transfer to coding. So the whole environment has to be nameable.
    """
    out: list[str] = []
    for part in (x.strip() for x in spec.split(",") if x.strip()):
        if part in ENVS:
            out += list(ENVS[part])
        elif part in COHORTS:
            out.append(part)
        else:
            raise SystemExit(f"unknown cohort or environment: {part!r}")
    return tuple(dict.fromkeys(out))


def args_have_policy(values: list[dict]) -> bool:
    """True when the controller head ran, so its policy is there to carry through."""
    return bool(values) and "policy" in values[0]


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
    strata = {k: int(rows[episodes[k][0]]["label"] > 0.5) for k in keys}  # median-ish split
    rng = np.random.RandomState(seed)
    folds: list[list[int]] = [[] for _ in range(FOLDS)]
    for value in (0, 1):
        members = [k for k in keys if strata[k] == value]
        rng.shuffle(members)
        for position, key in enumerate(members):
            folds[position % FOLDS] += episodes[key]
    return folds


class Head:
    """Six outputs: the three facts we can supervise, then the controller.

    0 success, 1 repeats, 2 stale -- supervised against labels.
    3 continue, 4 ask, 5 stop -- a policy, supervised against realised utility.

    Three actions, not the four the released schema names, because ``retry`` and
    ``continue`` have the same utility on a recorded trajectory: the agent cannot
    be made to restart, so the honest reading of either answer is that the
    episode runs on. Splitting probability between two identical columns would be
    arbitrary. When comparing against a released model's own choice we fold its
    ``retry`` into ``continue``, which is what gate_choice does.
    """

    OUTPUTS = 6
    ACTIONS = ("continue", "ask", "stop")


def build(model_id: str, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The question sits at the end of the prompt, and the coding prefixes run to
    # 10k characters. Truncating from the right -- the default -- would cut the
    # question off and leave the model guessing what it was asked. Cutting from
    # the left drops the oldest part of the issue text instead.
    tokenizer.truncation_side = "left"
    backbone = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=args.device)
    if args.lora_rank:
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        backbone.print_trainable_parameters()
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    return tokenizer, backbone


def make_head(width: int, args):
    """Sized from a real forward pass. Reading it from the config is not portable:
    Qwen3_5Config has no ``hidden_size`` and the first run died on exactly that."""
    import torch
    head = torch.nn.Sequential(
        torch.nn.Linear(width, args.hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, Head.OUTPUTS)).to(args.device).to(torch.float32)
    return head


def pool(backbone, tokenizer, texts: list[str], args):
    import torch
    encoded = tokenizer(texts, truncation=True, max_length=args.max_length, padding=True,
                        return_tensors="pt").to(args.device)
    states = backbone(**encoded).last_hidden_state
    index = encoded["attention_mask"].sum(1) - 1
    return states[torch.arange(states.shape[0], device=states.device), index].to(torch.float32)


def train_fold(rows: list[dict], train_idx, eval_idx, args) -> list[dict]:
    import torch

    tokenizer, backbone = build(args.model, args)
    head = make_head(pool(backbone, tokenizer, [rows[int(train_idx[0])]["text"]], args).shape[-1],
                     args)
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
            logits = head(pool(backbone, tokenizer, texts, args))
            device = logits.device
            y = torch.tensor([rows[i]["label"] for i in batch], dtype=torch.float32, device=device)
            q = torch.tensor([rows[i]["posterior"] for i in batch], dtype=torch.float32, device=device)
            p = torch.sigmoid(logits[:, 0])
            loss = args.w_outcome * bce(logits[:, 0], y) + args.w_brier * ((p - y) ** 2).mean()
            if args.w_distill:
                loss = loss + args.w_distill * ((p - q) ** 2).mean()
            if args.w_action:
                u = torch.tensor([utilities(rows[i], args.c_step, args.c_ask) for i in batch],
                                 dtype=torch.float32, device=device)
                policy = torch.softmax(logits[:, 3:6], dim=1)
                # maximise expected utility; the entropy bonus keeps the policy
                # from collapsing onto one column before it has seen the costs
                expected = (policy * u).sum(1).mean()
                entropy = -(policy * policy.clamp_min(1e-9).log()).sum(1).mean()
                loss = loss - args.w_action * expected - args.w_entropy * entropy
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
            logits = head(pool(backbone, tokenizer, [rows[j]["text"] for j in chunk], args))
            probabilities = torch.sigmoid(logits).float().cpu().tolist()
            policies = torch.softmax(logits[:, 3:6], dim=1).float().cpu().tolist()
            for j, values, policy in zip(chunk, probabilities, policies):
                out.append({"index": j, "success": values[0], "stuck": values[1],
                            "stale": values[2],
                            "policy": dict(zip(Head.ACTIONS, policy)),
                            "action": Head.ACTIONS[max(range(3), key=policy.__getitem__)]})
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
    p.add_argument("--max-length", type=int, default=2048,
                   help="coding prefixes average 6.4k characters and reach 10k, so 1536 cut most "
                        "of them; with left truncation the question always survives")
    p.add_argument("--w-action", type=float, default=0.0,
                   help="weight on expected realised utility: trains the controller head")
    p.add_argument("--w-entropy", type=float, default=0.01)
    p.add_argument("--c-step", type=float, default=0.02, help="as in gate_utility")
    p.add_argument("--c-ask", type=float, default=0.5, help="as in gate_utility")
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
    p.add_argument("--holdout", default=None,
                   help="cohorts to keep out of training and score instead: a cohort name, an "
                        "environment (alfworld, deepswe), or a comma-separated list. The closest "
                        "thing to a zero-shot claim this data allows. Pooled folds are "
                        "in-distribution -- a held-out episode's own cohort is still in training -- "
                        "and a fitted readout is a ceiling measurement, not something a reader "
                        "could deploy")
    p.add_argument("--drop", default=None,
                   help="cohorts to discard entirely, neither trained on nor scored. Use it to keep "
                        "a cross-cohort test inside one environment, so the result is not the "
                        "two-environment interference measured elsewhere")
    p.add_argument("--task-only", action="store_true",
                   help="drop the transcript and keep one row per episode: how much of an early "
                        "estimate is task difficulty rather than watching the agent")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="the control: permute within cohort, which must score at zero")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = load(a.views, a.targets, a.window, a.obs_chars, a.act_chars, a.task_only)
    if a.drop:
        dropped = cohort_spec(a.drop)
        before = len(rows)
        rows = [r for r in rows if r["cohort"] not in dropped]
        print(f"[train] dropped {','.join(dropped)}: {before} -> {len(rows)} prefixes", flush=True)
    if not rows:
        raise SystemExit("no prefixes: check --views and --targets")
    if a.shuffle_labels:
        rows = permute(rows)

    predictions: dict[int, list[dict]] = defaultdict(list)
    if a.holdout:
        held_cohorts = cohort_spec(a.holdout)
        train_idx = [i for i, r in enumerate(rows) if r["cohort"] not in held_cohorts]
        eval_idx = [i for i, r in enumerate(rows) if r["cohort"] in held_cohorts]
        if not train_idx or not eval_idx:
            raise SystemExit(f"--holdout {a.holdout}: {len(train_idx)} train and {len(eval_idx)} "
                             f"eval prefixes, one side is empty")
        print(f"[train] holding out {','.join(held_cohorts)}: {len(train_idx)} train prefixes "
              f"from {sorted({rows[i]['cohort'] for i in train_idx})}, "
              f"{len(eval_idx)} eval from {sorted({rows[i]['cohort'] for i in eval_idx})}",
              flush=True)
        for row in train_fold(rows, train_idx, eval_idx, a):
            predictions[row["index"]].append(row)
    else:
        for seed in a.seeds:
            for number, fold in enumerate(folds_by_episode(rows, seed), 1):
                held = set(fold)
                train_idx = [i for i in range(len(rows)) if i not in held]
                print(f"[train] seed {seed} fold {number}/{FOLDS}: "
                      f"{len(train_idx)} train, {len(fold)} eval", flush=True)
                for row in train_fold(rows, train_idx, fold, a):
                    predictions[row["index"]].append(row)

    tag = f"{a.tag}_shuffled" if a.shuffle_labels else a.tag
    if a.holdout:
        tag = f"{tag}_holdout{a.holdout.replace(',', '-')}"
    if a.task_only:
        tag = f"{tag}_taskonly"
    name = a.model.split("/")[-1]
    per_cohort: dict[str, list] = defaultdict(list)
    for index, values in predictions.items():
        row = rows[index]
        mean = {k: st.fmean(v[k] for v in values) for k in ("success", "stuck", "stale")}
        if args_have_policy(values):
            # average the policy over folds the same way, then take the action from
            # the averaged policy -- a per-fold argmax would throw the margins away
            mean["policy"] = {a: st.fmean(v["policy"][a] for v in values) for a in Head.ACTIONS}
            mean["action"] = max(Head.ACTIONS, key=mean["policy"].__getitem__)
        per_cohort[row["cohort"]].append((row, mean))
    for cohort, items in per_cohort.items():
        with open(a.out / f"steps_{tag}_{name}_{cohort}.jsonl", "w") as handle:
            for row, value in sorted(items, key=lambda t: (t[0]["episode"], t[0]["step"])):
                record = {"id": row["episode"], "step": row["step"],
                          "p": value["success"], "stuck": value["stuck"],
                          "stale": value["stale"]}
                if "policy" in value:
                    # named "choice" and "p_<action>" so gate_choice scores this file
                    # exactly as it scores a released model's own answers: same
                    # accounting, same references, no second scorer to keep honest
                    record["choice"] = value["action"]
                    record.update({f"p_{a}": value["policy"][a] for a in Head.ACTIONS})
                handle.write(json.dumps(record) + "\n")
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
