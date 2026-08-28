#!/usr/bin/env python3
"""Вытащить код каждой генерации из сайдкара логпробов и закэшировать.

Сайдкары по 3.5-8.8 ГБ, но нужен только `code`, поэтому парсим построчно и
кладём {(instance_id, generation_index): code} в pickle.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import json, pickle, sys
from pathlib import Path

root, stem, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
codes = {}
with (root / f"{stem}.generation_logprobs.jsonl").open() as fh:
    for line in fh:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = r.get("code")
        if c is not None:
            codes[(str(r.get("instance_id")), int(r.get("generation_index", 0)))] = c
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("wb") as fh:
    pickle.dump(codes, fh)
print(f"{stem}: {len(codes)} кандидатов -> {out}")
