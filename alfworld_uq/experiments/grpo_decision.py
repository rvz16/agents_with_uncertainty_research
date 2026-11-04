"""GRPO over the confidence head: the decision, not the probability.

Why a policy gradient at all. ``train_decision`` argues, correctly, that a
controller does not need one: every prefix comes from a finished recording, so
the payoff of continuing, asking and stopping is *known*, and expected utility
under a stochastic policy is then an exact differentiable function of its
probabilities. That argument holds for a policy over actions. It stops holding
the moment the thing we ship is a *number* and the action is read off that
number by a threshold the deployment picks: the map from confidence to action is
a step function, its gradient is zero almost everywhere, and a proper scoring
rule on the number is a proxy for the decision rather than the decision itself.
So the question GRPO answers here is narrow and worth asking: trained against
the realised utility of the decision its confidence causes, does the head rank
and calibrate better than one trained against log loss plus Brier?

What is borrowed from CALIBER (arXiv 2606.24281) and what is not. Borrowed: the
group-relative advantage with *no* standard-deviation normalisation, and no KL
to a reference policy -- both of their ablations say the normalisation is what
makes confidence training collapse onto one value. Not borrowed: their reward
adds a squared-error term on the confidence to an accuracy term on the answer,
which is a weighted Brier score wearing a reward's clothes. Ours is the
realised gate utility, with the calibration term kept only as a small shaping
bonus (``--w-cal``), so what the policy maximises is what the paper's own gate
table scores.

The policy. A categorical over 21 confidence bins, 0.00 to 1.00, in place of
one sigmoid. Sampling is free: one forward pass gives the whole distribution,
so a group of G confidences costs no more than a single supervised step, which
is why this runs in an hour where the SFT recipe runs in ten. Every sample is
drawn from the current policy and used once, so there is no stale-sample
correction to make: the plain score-function estimator is unbiased and the PPO
ratio and its clip would both be identically one.

Costs. ``--c-step`` as a fixed 0.02 is degenerate on this data -- stopping at
step zero beats letting the episode run in 87% of DG episodes and 97% of DQ --
so the step price is set as ``--kappa`` divided by the median episode length,
which prices a whole median-length episode at kappa whatever the cohort's
horizon. At the default 0.3 continuing a success pays 0.7, continuing a failure
pays -0.3, asking pays about 0.48 and stopping early pays about -0.02: four
distinguishable payoffs, which is the least a reward needs.

Data. DG and DQ only, by default: they are the two cohorts with *verified*
per-prefix labels, and a reward computed from a label forward-filled from the
end of the episode would be a reward for hindsight.

Three arms make the comparison: SFT alone (``train_decision`` on the same
cohorts), SFT then GRPO (``--init-adapter``), and GRPO from the base model.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

import experiments.train_decision as td
from experiments.train_decision import (COHORTS, auroc, cohort_spec, folds_by_episode, load,
                                        pool, report, utilities)

#: 21 bins, 0.00 .. 1.00. Fewer cannot express a calibrated 0.37; many more make
#: the per-bin sample count in a group of eight meaningless.
BINS = 21
VALUES = tuple(i / (BINS - 1) for i in range(BINS))


def price(rows: list[dict], kappa: float) -> float:
    """The per-step cost that prices a median-length episode at ``kappa``."""
    median = st.median([r["n_steps"] for r in rows]) or 1.0
    return kappa / float(median)


def action_of(value: float, tau_stop: float, tau_ask: float) -> str:
    """The deployment's rule, not the model's: a threshold on the number we ship.

    This is the function whose zero gradient is the whole reason for a policy
    gradient here. The two thresholds are the gate's, and they are held fixed
    during training -- a policy that could move them would be optimising the
    scorer instead of the decision.
    """
    if value < tau_stop:
        return "stop"
    if value < tau_ask:
        return "ask"
    return "continue"


def reward(row: dict, value: float, c_step: float, c_ask: float,
           tau_stop: float, tau_ask: float, w_cal: float) -> float:
    payoff = dict(zip(("continue", "ask", "stop"), utilities(row, c_step, c_ask)))
    gate = payoff[action_of(value, tau_stop, tau_ask)]
    return gate + w_cal * (1.0 - (value - row["label"]) ** 2)


def make_head(width: int, args):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(width, args.hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, BINS)).to(args.device).to(torch.float32)


def fold_adapter(root, fold_tag: str):
    """The supervised adapter for *this* fold, or nothing.

    Per fold, not one adapter for all of them. The supervised run cut its folds
    from the same rows with the same seed, so its fold 1 adapter saw exactly the
    episodes this fold 1 trains on; borrowing a different fold's adapter would
    put the evaluation episodes into training through the weights, which is the
    quiet version of the leak this project keeps finding in other people's
    numbers.
    """
    if not root:
        return None
    path = Path(root) / (fold_tag.strip("_") or "fold")
    if not path.is_dir():
        raise SystemExit(f"--init-adapter {root}: no adapter for this fold at {path}. The "
                         f"supervised run must have been trained with the same --folds and seed")
    return path


def build(args, adapter):
    """The backbone, with the SFT adapter loaded when there is one to load.

    ``--init-adapter`` is the arm that matters: the same LoRA the supervised run
    finished with, continued under the reward. Without it the adapter starts
    random, which is the from-scratch arm.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    if str(args.device).startswith("mps"):
        backbone = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    else:
        backbone = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                             device_map=args.device)
    if adapter:
        backbone = PeftModel.from_pretrained(backbone, str(adapter), is_trainable=True)
        print(f"[grpo] adapter loaded from {adapter}", flush=True)
    else:
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    backbone.print_trainable_parameters()
    return tokenizer, backbone


def sft_probability(args, adapter, width: int):
    """The supervised head, frozen, only to warm-start the bins.

    A fresh categorical head on a trained trunk would throw the supervised
    answer away on its first step, and the arm would stop being "GRPO on top of
    SFT". So the bins are first fitted to the probability the supervised head
    gives -- a soft target, a Gaussian over bin centres -- and only then does
    the reward take over.
    """
    import torch
    path = Path(adapter) / "head.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    head = torch.nn.Sequential(
        torch.nn.Linear(width, state.get("hidden", args.hidden)), torch.nn.GELU(),
        torch.nn.Dropout(0.0), torch.nn.Linear(state.get("hidden", args.hidden), 6))
    head.load_state_dict(state["head"])
    head = head.to(args.device).to(torch.float32).eval()
    for parameter in head.parameters():
        parameter.requires_grad = False
    print(f"[grpo] supervised head loaded from {path}", flush=True)
    return head


def soft_bins(targets, sigma: float, device):
    """A Gaussian over bin centres: the supervised scalar as a distribution."""
    import torch
    centres = torch.tensor(VALUES, dtype=torch.float32, device=device)
    weight = torch.exp(-((centres[None, :] - targets[:, None]) ** 2) / (2 * sigma ** 2))
    return weight / weight.sum(1, keepdim=True).clamp_min(1e-9)


def expected(logits):
    import torch
    centres = torch.tensor(VALUES, dtype=torch.float32, device=logits.device)
    return (torch.softmax(logits, dim=1) * centres).sum(1)


def evaluate(backbone, head, tokenizer, rows, eval_idx, args, c_step):
    """What the arm is judged on, all of it off the same forward pass.

    Three numbers, because they can disagree and the disagreement is the result:
    AUROC of the expected confidence (ranking), its Brier score (calibration),
    and the realised utility of the action that confidence causes (the decision).
    """
    import torch
    backbone.eval(); head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(eval_idx), args.eval_batch):
            chunk = list(eval_idx)[i: i + args.eval_batch]
            logits = head(pool(backbone, tokenizer, [rows[j]["text"] for j in chunk], args))
            values = expected(logits).float().cpu().tolist()
            entropy = (-(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1)
                       .float().cpu().tolist())
            for j, value, h in zip(chunk, values, entropy):
                row = rows[j]
                choice = action_of(value, args.tau_stop, args.tau_ask)
                payoff = dict(zip(("continue", "ask", "stop"),
                                  utilities(row, c_step, args.c_ask)))
                out.append({"index": j, "p": value, "entropy": h, "choice": choice,
                            "utility": payoff[choice], "label": row["label"]})
    return out


def train_fold(rows, train_idx, eval_idx, args, c_step, fold_tag: str):
    import torch

    adapter = fold_adapter(args.init_adapter, fold_tag)
    tokenizer, backbone = build(args, adapter)
    width = pool(backbone, tokenizer, [rows[int(train_idx[0])]["text"]], args).shape[-1]
    head = make_head(width, args)
    reference = sft_probability(args, adapter, width) if adapter else None

    parameters = [p for p in list(backbone.parameters()) + list(head.parameters())
                  if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    order = np.random.RandomState(0).permutation(list(train_idx))
    batches = [order[i: i + args.batch] for i in range(0, len(order), args.batch)]
    total = max(1, len(batches) * args.epochs + args.warmup)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=total,
                                                   pct_start=0.1, anneal_strategy="linear")
    backbone.train(); head.train()

    # Warm-start: fit the bins to the supervised probability before the reward
    # touches them. Skipped when there is no supervised head to copy.
    if reference is not None and args.warmup:
        for step in range(1, args.warmup + 1):
            batch = batches[(step - 1) % len(batches)]
            features = pool(backbone, tokenizer, [rows[i]["text"] for i in batch], args)
            with torch.no_grad():
                target = torch.sigmoid(reference(features)[:, 0])
            loss = -(soft_bins(target, args.sigma, features.device)
                     * torch.log_softmax(head(features), dim=1)).sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print(f"[grpo] warmup {step}/{args.warmup} ce {float(loss):.4f}", flush=True)

    for epoch in range(args.epochs):
        for number, batch in enumerate(batches, 1):
            logits = head(pool(backbone, tokenizer, [rows[i]["text"] for i in batch], args))
            device = logits.device
            logp = torch.log_softmax(logits, dim=1)
            with torch.no_grad():
                drawn = torch.multinomial(logp.exp(), args.group, replacement=True)
            values = torch.tensor(VALUES, device=device)[drawn]
            rewards = torch.tensor(
                [[reward(rows[i], float(v), c_step, args.c_ask, args.tau_stop, args.tau_ask,
                         args.w_cal) for v in row] for i, row in zip(batch, values.cpu().tolist())],
                dtype=torch.float32, device=device)
            # group-relative, and deliberately not divided by the group's standard
            # deviation: on a prefix the policy is already sure about, the spread
            # is near zero and that division turns noise into a large gradient --
            # the collapse CALIBER's own ablation reports.
            advantage = rewards - rewards.mean(dim=1, keepdim=True)
            chosen = logp.gather(1, drawn)
            objective = (advantage * chosen).mean()
            entropy = -(logp.exp() * logp).sum(1).mean()
            loss = -objective - args.w_entropy * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
            if number % 25 == 0:
                step = epoch * len(batches) + number
                print(f"[grpo] epoch {epoch + 1} batch {number}/{len(batches)} "
                      f"reward {float(rewards.mean()):.4f} entropy {float(entropy):.3f} "
                      f"p {float(expected(logits).mean()):.3f}", flush=True)
                report(f"reward{fold_tag}", "mean", float(rewards.mean()), step)
                report(f"reward{fold_tag}", "spread", float(rewards.std()), step)
                report(f"policy{fold_tag}", "entropy", float(entropy), step)
                report(f"policy{fold_tag}", "mean_p", float(expected(logits).mean()), step)
        if args.eval_each_epoch:
            scored = evaluate(backbone, head, tokenizer, rows, eval_idx, args, c_step)
            value = auroc([s["label"] for s in scored], [s["p"] for s in scored])
            brier = st.fmean((s["p"] - s["label"]) ** 2 for s in scored)
            gained = st.fmean(s["utility"] for s in scored)
            report(f"eval{fold_tag}", "auroc", value, epoch + 1)
            report(f"eval{fold_tag}", "brier", brier, epoch + 1)
            report(f"eval{fold_tag}", "utility", gained, epoch + 1)
            print(f"[grpo] epoch {epoch + 1} eval: AUROC {value:.3f} brier {brier:.3f} "
                  f"utility {gained:+.4f} mean p {st.fmean(s['p'] for s in scored):.3f}",
                  flush=True)
            backbone.train(); head.train()

    scored = evaluate(backbone, head, tokenizer, rows, eval_idx, args, c_step)
    if args.save_weights:
        target = Path(args.save_weights) / (fold_tag.strip("_") or "fold")
        target.mkdir(parents=True, exist_ok=True)
        try:
            backbone.save_pretrained(target)
        except Exception as error:
            print(f"[grpo] adapter not saved: {error}", flush=True)
        torch.save({"head": head.state_dict(), "bins": list(VALUES), "model": args.model,
                    "hidden": args.hidden, "c_step": c_step}, target / "head.pt")
        try:
            from clearml import Task
            task = Task.current_task()
            if task:
                task.upload_artifact(f"weights_{target.name}", artifact_object=target)
        except Exception as error:
            print(f"[grpo] weights not uploaded: {error}", flush=True)
    del backbone, head
    torch.cuda.empty_cache()
    return scored


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--targets", type=Path, default=Path("data/distill_targets"))
    p.add_argument("--out", type=Path, default=Path("data/grpo_runs"))
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", default="grpo")
    p.add_argument("--keep", default="DG,DQ",
                   help="cohorts to train and score on. The default is the pair with verified "
                        "per-prefix labels: a reward read off a forward-filled label is a reward "
                        "for hindsight")
    p.add_argument("--init-adapter", default="",
                   help="the supervised run's weights root, holding one adapter directory per "
                        "fold (s0f1, s0f2, ...). With it this is the on-top-of-SFT arm, without "
                        "it the from-scratch one")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--group", type=int, default=8, help="G: confidences sampled per prefix")
    p.add_argument("--warmup", type=int, default=100,
                   help="steps fitting the bins to the supervised probability before the reward "
                        "takes over; ignored without --init-adapter")
    p.add_argument("--sigma", type=float, default=0.05, help="width of that soft bin target")
    p.add_argument("--w-entropy", type=float, default=0.01)
    p.add_argument("--w-cal", type=float, default=0.3,
                   help="weight on the calibration bonus. The gate utility alone rewards only the "
                        "side of the threshold the number lands on, so a little Brier keeps the "
                        "number itself meaningful. Set 0 for the pure decision reward")
    p.add_argument("--kappa", type=float, default=0.3,
                   help="what a whole median-length episode costs, from which the per-step price "
                        "follows. Fixes the degeneracy of a flat --c-step across cohorts whose "
                        "horizons differ by an order of magnitude")
    p.add_argument("--c-step", type=float, default=0.0,
                   help="a flat per-step price, overriding --kappa when non-zero. Only for "
                        "reproducing the degenerate setting on purpose")
    p.add_argument("--c-ask", type=float, default=0.5)
    p.add_argument("--tau-stop", type=float, default=0.0,
                   help="the band below which the confidence stops the episode. Zero by default, "
                        "and the reason is structural rather than empirical: asking and stopping "
                        "differ by exactly 1 - c_ask at every prefix, so whichever is larger "
                        "dominates the other everywhere, and at c_ask below 1 stopping is never "
                        "the right answer. A stop band would then punish the policy for putting "
                        "low confidence on an episode that really does fail -- the one thing we "
                        "want it to do. Raise both this and --c-ask past 1 to train the halt gate "
                        "instead")
    p.add_argument("--tau-ask", type=float, default=0.5,
                   help="the band below which the confidence calls a human. The deployment's "
                        "threshold, fixed during training")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--obs-chars", type=int, default=600)
    p.add_argument("--act-chars", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-limit", type=int, default=0)
    p.add_argument("--eval-each-epoch", type=int, default=1)
    p.add_argument("--save-weights", default="")
    p.add_argument("--row-limit", type=int, default=0, help="a smoke test, not a result")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    td.FOLDS = a.folds
    rows = load(a.views, a.targets, a.window, a.obs_chars, a.act_chars)
    keep = cohort_spec(a.keep)
    rows = [r for r in rows if r["cohort"] in keep]
    if not rows:
        raise SystemExit(f"no prefixes left after --keep {a.keep}")
    if a.row_limit:
        # evenly spaced, not the first N: the first N prefixes are one episode,
        # which leaves a fold split with nothing to train on
        rows = rows[:: max(1, len(rows) // a.row_limit)][: a.row_limit]
    c_step = a.c_step if a.c_step else price(rows, a.kappa)
    print(f"[grpo] {len(rows)} prefixes over {len({(r['cohort'], r['episode']) for r in rows})} "
          f"episodes from {','.join(keep)}; c_step {c_step:.5f} "
          f"(median length {st.median([r['n_steps'] for r in rows]):.0f})", flush=True)
    # What the reward can distinguish, printed before training rather than hoped
    # for: a cost model that makes one action dominate everywhere is the failure
    # this project already walked into once at a flat c_step of 0.02.
    best = defaultdict(int)
    for row in rows:
        payoff = dict(zip(("continue", "ask", "stop"), utilities(row, c_step, a.c_ask)))
        best[max(payoff, key=payoff.__getitem__)] += 1
    print(f"[grpo] best action under this cost model: "
          f"{dict(sorted(best.items(), key=lambda kv: -kv[1]))}", flush=True)
    if a.c_ask < 1.0 and a.tau_stop > 0.0:
        print("[grpo] warning: at c_ask < 1 asking dominates stopping on every prefix, so the "
              "stop band rewards the policy for not being confidently pessimistic. Set "
              "--tau-stop 0, or price the human above the task", flush=True)
    if len(best) < 2:
        raise SystemExit(f"one action is best on all {len(rows)} prefixes: nothing for a reward "
                         f"to separate. Change --kappa or --c-ask")

    predictions: dict[int, list[dict]] = defaultdict(list)
    for seed in a.seeds:
        for number, fold in enumerate(folds_by_episode(rows, seed), 1):
            if a.fold_limit and number > a.fold_limit:
                break
            held = set(fold)
            train_idx = [i for i in range(len(rows)) if i not in held]
            if not train_idx or not fold:
                continue
            print(f"[grpo] seed {seed} fold {number}/{a.folds}: {len(train_idx)} train, "
                  f"{len(fold)} eval", flush=True)
            for scored in train_fold(rows, train_idx, fold, a, c_step, f"_s{seed}f{number}"):
                predictions[scored["index"]].append(scored)

    name = a.model.split("/")[-1]
    tag = f"{a.tag}_sft" if a.init_adapter else a.tag
    per_cohort: dict[str, list] = defaultdict(list)
    for index, values in predictions.items():
        row = rows[index]
        mean = {"p": st.fmean(v["p"] for v in values),
                "entropy": st.fmean(v["entropy"] for v in values)}
        mean["choice"] = action_of(mean["p"], a.tau_stop, a.tau_ask)
        payoff = dict(zip(("continue", "ask", "stop"), utilities(row, c_step, a.c_ask)))
        mean["utility"] = payoff[mean["choice"]]
        per_cohort[row["cohort"]].append((row, mean))
    summary = {}
    for cohort, items in per_cohort.items():
        # the same schema train_decision writes, so prefix_uq and gate_choice
        # score this arm with no second scorer to keep honest
        with open(a.out / f"steps_{tag}_{name}_{cohort}.jsonl", "w") as handle:
            for row, value in sorted(items, key=lambda t: (t[0]["episode"], t[0]["step"])):
                handle.write(json.dumps({"id": row["episode"], "step": row["step"],
                                         "p": value["p"], "choice": value["choice"],
                                         "entropy": value["entropy"]}) + "\n")
        last: dict[str, tuple] = {}
        for row, value in items:
            if row["episode"] not in last or row["step"] > last[row["episode"]][0]["step"]:
                last[row["episode"]] = (row, value)
        with open(a.out / f"answers_{tag}_{name}_{cohort}.jsonl", "w") as handle:
            for episode, (row, value) in sorted(last.items()):
                handle.write(json.dumps({
                    "id": episode, "cohort": cohort, "label": int(row["label"] > 0.5),
                    "score": row["label"],
                    "answer": {"model": f"{a.model}+lora+grpo", "answers": {
                        "success": {"type": "noul", "noul": value["p"]}}}}) + "\n")
        summary[cohort] = {
            "prefixes": len(items), "episodes": len(last),
            "auroc": auroc([r["label"] for r, _ in items], [v["p"] for _, v in items]),
            "brier": st.fmean((v["p"] - r["label"]) ** 2 for r, v in items),
            "utility": st.fmean(v["utility"] for _, v in items),
            "mean_p": st.fmean(v["p"] for _, v in items),
            "mean_label": st.fmean(r["label"] for r, _ in items),
            "entropy": st.fmean(v["entropy"] for _, v in items),
            "choices": dict(sorted(
                ((k, sum(1 for _, v in items if v["choice"] == k)) for k in
                 ("continue", "ask", "stop")), key=lambda kv: -kv[1])),
        }
        print(f"[grpo] {cohort}: AUROC {summary[cohort]['auroc']:.3f} "
              f"brier {summary[cohort]['brier']:.3f} "
              f"utility {summary[cohort]['utility']:+.4f} "
              f"choices {summary[cohort]['choices']}", flush=True)
    (a.out / f"summary_{tag}_{name}.json").write_text(json.dumps(summary, indent=2))
    print("[grpo] done")


if __name__ == "__main__":
    main()
