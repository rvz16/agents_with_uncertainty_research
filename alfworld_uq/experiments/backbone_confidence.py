"""The backbone's own answer, for the ablation against the decision model.

Clef is a frozen Qwen with a trained joint schema head on top. The question this
answers is what that head buys: the same backbone, the same state, the same
question, but the probability read from the language-model head instead.

Two readings, both from one forward pass per state:

* ``token``  -- P(yes) against P(no) at the next position, renormalised over the
  two. This is the closest analogue of a noul and needs no parsing.
* ``verbal`` -- the model is asked for an integer 0-100 and it is generated.
  This is what our prefix judge does, kept here so the two are comparable.

Writes the answer file the scorer already reads, with the probability under
``answers.success``, so a backbone row sits beside a decision-model row.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

QUESTION = ("Will this agent ultimately complete the task correctly? "
            "The trajectory is cut before its final action. Answer yes or no.")
VERBAL = ("Will this agent ultimately complete the task correctly? The trajectory is cut before its "
          "final action. Reply with a single integer from 0 to 100: the probability that it succeeds.")
NUM = re.compile(r"(\d{1,3})")


def build(tokenizer, state: str, question: str) -> str:
    messages = [{"role": "system", "content": "You are an evaluator of an autonomous agent's trajectory. "
                                              "You do not continue the agent's work."},
                {"role": "user", "content": f"{state}\n\n{question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    def ids(words):
        out = []
        for w in words:
            for form in (w, " " + w):
                token = tokenizer.encode(form, add_special_tokens=False)
                if len(token) == 1:
                    out.append(token[0])
        return sorted(set(out))
    return ids(["yes", "Yes", "YES"]), ids(["no", "No", "NO"])


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--mode", choices=["token", "verbal"], default="token")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device).eval()
    yes, no = yes_no_ids(tokenizer)
    print(f"[backbone] yes ids {yes}, no ids {no}", flush=True)

    done: set[tuple[str, str]] = set()
    if a.out.exists():
        kept = [line for line in open(a.out)
                if line.strip() and "error" not in json.loads(line)["answer"]]
        with open(a.out, "w") as handle:
            handle.writelines(kept)
        done = {(json.loads(line)["cohort"], json.loads(line)["id"]) for line in kept}
        print(f"[backbone] resuming: {len(done)} usable answers kept", flush=True)

    rows = []
    for path in a.states:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if (row["cohort"], row["id"]) not in done:
                    rows.append(row)
    rows = rows[: a.limit or None]
    print(f"[backbone] {len(rows)} states to score with {a.model} ({a.mode})", flush=True)

    out = open(a.out, "a")
    started = time.time()
    for index, row in enumerate(rows, 1):
        text = build(tokenizer, row["state"], QUESTION if a.mode == "token" else VERBAL)
        batch = tokenizer(text, return_tensors="pt").to(a.device)
        try:
            if a.mode == "token":
                with torch.no_grad():
                    logits = model(**batch).logits[0, -1].float()
                probabilities = torch.softmax(logits, dim=-1)
                py = float(probabilities[yes].sum()); pn = float(probabilities[no].sum())
                value = py / (py + pn) if (py + pn) > 0 else None
                answer = {"model": a.model, "answers": {"success": {"type": "noul", "noul": value}},
                          "p_yes_raw": py, "p_no_raw": pn}
            else:
                with torch.no_grad():
                    generated = model.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=False)
                text_out = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                found = NUM.findall(text_out)
                value = min(max(int(found[-1]), 0), 100) / 100 if found else None
                answer = {"model": a.model, "answers": {"success": {"type": "noul", "noul": value}},
                          "text": text_out[:120]}
            if value is None:
                answer = {"error": "no probability", **answer}
            answer["latency_ms"] = 0.0
        except Exception as exc:  # noqa: BLE001
            answer = {"error": repr(exc)}
        if index == 1:
            print("[backbone] first answer:", json.dumps(answer)[:300], flush=True)
        out.write(json.dumps({"id": row["id"], "cohort": row["cohort"], "label": row["label"],
                              "score": row["score"], "answer": answer}) + "\n")
        if index % 50 == 0:
            out.flush()
            print(f"[backbone] {index}/{len(rows)}  {index / max(time.time() - started, 1e-6):.2f}/s", flush=True)
    out.close()
    print(f"[backbone] done -> {a.out}")


if __name__ == "__main__":
    main()
