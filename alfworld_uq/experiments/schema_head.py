"""Train the decision head ourselves, on a frozen backbone.

The appendix shows that a released decision head does not beat the backbone it
was built on, and that a 149M encoder fine-tuned on our episodes beats the head.
Neither result says the head is a bad mechanism: the first compares a head
trained on someone else's distribution, the second changes the backbone as well
as the training data. The missing third point is a schema head trained on our
own episodes over the same frozen backbone the released model uses, which is
what this builds.

It runs in two stages so the expensive half happens once. Stage one reads every
state through a frozen Qwen with no gradients and keeps two vectors per state --
the final position's hidden state and the mean over positions -- which is the
representation a schema head sits on. Stage two trains the head on those
vectors under the same folds as every other estimator, three seeds by five
folds, so nothing is scored by a head that saw it.

Three heads are trained on identical features, which is the point: a logistic
regression, to measure the features alone; a small MLP under plain
cross-entropy; and the same MLP under the objective the released models
describe, a label-smoothed cross-entropy with a Brier term. If the head is the
mechanism, the third beats the first. If the features are, they tie.

With ``--aux`` the MLP also predicts two structural facts for which ALFWorld
gives mechanical labels -- whether the episode ever repeated an action and
whether it ever acted on stale evidence -- which is a genuine multi-question
schema rather than one binary dressed up as one.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
FOLDS = 5
COHORTS = ("RG", "SG", "RQ", "SQ")
CHECK = ("look", "examine", "inventory", "go to")
MUTATE = ("take ", "put ", "open ", "close ", "toggle ", "heat ", "cool ", "clean ", "slice ", "use ")


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def structural_labels(steps: list[dict]) -> tuple[float, float]:
    """The share of steps that repeat, and the share taken on stale evidence.

    "Did it ever" is the wrong target: on these cohorts 96% of episodes contain
    at least one stale step, so the binary version is a constant and teaches the
    head nothing. The share is informative for both facts and still a
    probability, so it trains under the same cross-entropy.
    """
    seen: set[tuple[str, str]] = set()
    repeated = stale = 0
    last_check, mutated = -1, False
    for k, step in enumerate(steps):
        action, observation = norm(step.get("action")), norm(step.get("observation"))
        repeated += ((action, observation) in seen) or ("nothing happens" in observation)
        seen.add((action, observation))
        stale += last_check >= 0 and mutated
        if any(action.startswith(c) for c in CHECK):
            last_check, mutated = k, False
        elif any(action.startswith(m) for m in MUTATE):
            mutated = True
    n = max(len(steps), 1)
    return repeated / n, stale / n


def load_states(states: Path, views: Path | None) -> list[dict]:
    rows = []
    structural: dict[tuple[str, str], tuple[int, int]] = {}
    if views is not None:
        for path in sorted(views.glob("*.jsonl")):
            for line in open(path):
                if line.strip():
                    view = json.loads(line)
                    structural[(path.stem, view["id"])] = structural_labels(view["steps"])
    for path in sorted(states.glob("*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") is None:
                continue
            row["repeated"], row["stale"] = structural.get((row["cohort"], row["id"]), (0, 0))
            rows.append(row)
    return rows


def extract(rows: list[dict], model_id: str, device: str, max_length: int, batch: int) -> np.ndarray:
    """Two pooled vectors per state from a frozen backbone, concatenated."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
    out = []
    for start in range(0, len(rows), batch):
        texts = [r["state"] for r in rows[start: start + batch]]
        encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        last = hidden[torch.arange(hidden.shape[0]), encoded["attention_mask"].sum(1) - 1]
        out.append(torch.cat([last, mean], dim=-1).cpu().numpy().astype(np.float32))
        if (start // batch) % 10 == 0:
            print(f"[schema] features {start + len(texts)}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def folds_of(labels: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    folds: list[list[int]] = [[] for _ in range(FOLDS)]
    for value in (0, 1):
        members = np.where(labels == value)[0]
        rng.shuffle(members)
        for position, index in enumerate(members):
            folds[position % FOLDS].append(int(index))
    return [np.asarray(f) for f in folds]


def train_head(x: np.ndarray, y: np.ndarray, aux: np.ndarray | None, x_eval: np.ndarray,
               kind: str, args) -> np.ndarray:
    """``kind`` is 'logreg', 'mlp' or 'schema' (label smoothing plus a Brier term)."""
    import torch

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(max_iter=2000, C=args.inverse_penalty)
        model.fit(scaler.transform(x), y)
        return model.predict_proba(scaler.transform(x_eval))[:, 1]

    device = args.device
    mean, scale = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x - mean) / scale, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    outputs = 1 + (aux.shape[1] if aux is not None else 0)
    at = torch.tensor(aux, dtype=torch.float32, device=device) if aux is not None else None

    torch.manual_seed(0)
    head = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], args.hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, outputs)).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    smoothing = args.smoothing if kind == "schema" else 0.0
    for epoch in range(args.epochs):
        order = torch.randperm(xt.shape[0], device=device)
        for start in range(0, xt.shape[0], args.batch):
            index = order[start: start + args.batch]
            logits = head(xt[index])
            target = yt[index] * (1.0 - smoothing) + 0.5 * smoothing
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target)
            if kind == "schema":  # the released recipe pairs the smoothed loss with a Brier term
                loss = loss + args.brier * ((torch.sigmoid(logits[:, 0]) - yt[index]) ** 2).mean()
            if at is not None:
                loss = loss + args.aux_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 1:], at[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimiser.step(); optimiser.zero_grad(set_to_none=True)
    head.eval()
    with torch.no_grad():
        xe = torch.tensor((x_eval - mean) / scale, dtype=torch.float32, device=device)
        return torch.sigmoid(head(xe)[:, 0]).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, default=Path("data/decision_states_clean"))
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--features", type=Path, default=None,
                   help="reuse a cached .npy instead of a second pass over the backbone")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--feature-batch", type=int, default=4)
    p.add_argument("--kinds", nargs="+", default=["logreg", "mlp", "schema"])
    p.add_argument("--aux", action="store_true", help="also predict the two structural facts")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--brier", type=float, default=1.0)
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--inverse-penalty", type=float, default=0.1)
    p.add_argument("--shuffle-labels", action="store_true",
                   help="permute the labels within each cohort before the folds are cut. The control "
                        "for everything else here: a probe on 8192 features trained on some 350 "
                        "episodes can separate noise, so a high score is only evidence if this run "
                        "comes back at zero")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = load_states(a.states, a.views if a.aux else None)
    print(f"[schema] {len(rows)} states, "
          f"{ {c: sum(r['cohort'] == c for r in rows) for c in COHORTS} }", flush=True)

    cache = a.features or (a.out / f"features_{a.model.replace('/', '_')}.npy")
    if cache.exists():
        features = np.load(cache)
        print(f"[schema] reusing {cache} {features.shape}", flush=True)
    else:
        features = extract(rows, a.model, a.device, a.max_length, a.feature_batch)
        np.save(cache, features)
        print(f"[schema] wrote {cache} {features.shape}", flush=True)
    assert features.shape[0] == len(rows), (features.shape, len(rows))

    labels = np.asarray([int(r["label"]) for r in rows])
    if a.shuffle_labels:
        rng = np.random.RandomState(12345)
        for cohort in COHORTS:
            index = np.where(np.asarray([r["cohort"] for r in rows]) == cohort)[0]
            labels[index] = labels[rng.permutation(index)]
        print("[schema] labels permuted within cohort: this run is the control", flush=True)
    structural = np.asarray([[r["repeated"], r["stale"]] for r in rows], dtype=np.float32)
    print(f"[schema] auxiliary targets: repeat share {structural[:, 0].mean():.2f}, "
          f"stale share {structural[:, 1].mean():.2f}", flush=True)
    by_cohort: dict[str, np.ndarray] = {c: np.where(np.asarray([r["cohort"] for r in rows]) == c)[0]
                                        for c in COHORTS}

    for kind in a.kinds:
        name = f"{kind}+aux" if (a.aux and kind != "logreg") else kind
        predictions: dict[int, list[float]] = defaultdict(list)
        for cohort, index in by_cohort.items():
            x, y = features[index], labels[index]
            aux = structural[index] if (a.aux and kind != "logreg") else None
            for seed in SEEDS:
                for fold in folds_of(y, seed):
                    held = np.zeros(len(index), dtype=bool)
                    held[fold] = True
                    values = train_head(x[~held], y[~held], None if aux is None else aux[~held],
                                        x[held], kind, a)
                    for position, value in zip(index[held], values):
                        predictions[int(position)].append(float(value))
            print(f"[schema] {name} {cohort} done", flush=True)
        if a.shuffle_labels:
            name = f"{name}_shuffled"
        path = a.out / f"answers_schema_{name}_{a.model.split('/')[-1]}.jsonl"
        with open(path, "w") as handle:
            for position, row in enumerate(rows):
                if position not in predictions:
                    continue
                handle.write(json.dumps({
                    "id": row["id"], "cohort": row["cohort"], "label": row["label"],
                    "score": row["score"],  # always the real outcome, so a control scores against truth
                    "answer": {"model": f"{a.model} + {name} head",
                               "answers": {"success": {"type": "noul",
                                                       "noul": st.fmean(predictions[position])}}}}) + "\n")
        print(f"[schema] wrote {path}", flush=True)
    print("[schema] done")


if __name__ == "__main__":
    main()
