"""OSWorld task instructions by id -- the one thing the colleague's archive does not carry.

evaluation_examples/examples/<domain>/<task_id>.json in xlang-ai/OSWorld; the
domain comes from test_all.json, cached beside this file as osworld_index.json.
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/xlang-ai/OSWorld/main/evaluation_examples/examples/{}/{}.json"
index = json.loads((HERE / "osworld_index.json").read_text())
domain = {t: d for d, ts in index.items() for t in ts}
wanted = sorted({json.loads(l)["episode_id"]
                 for p in Path(os.environ.get("OSWORLD_RAW", HERE / "raw")).rglob("uq/episodes.jsonl")
                 for l in open(p) if l.strip()})

def one(task):
    try:
        with urlopen(RAW.format(domain[task], task), timeout=30) as r:
            d = json.load(r)
        return task, (d.get("instruction") or "").strip()
    except Exception as exc:                     # noqa: BLE001
        return task, f"__error__ {exc}"

with ThreadPoolExecutor(max_workers=6) as pool:
    got = dict(pool.map(one, wanted))
bad = {k: v for k, v in got.items() if v.startswith("__error__") or not v}
print(f"{len(got)} ids, {len(bad)} failed")
for k, v in list(bad.items())[:5]:
    print("  ", k, v[:90])
(HERE / "osworld_instructions.json").write_text(json.dumps(got, indent=0, ensure_ascii=False))
lens = sorted(len(v) for v in got.values() if not v.startswith("__error__"))
print(f"instruction chars: median {lens[len(lens)//2]}, max {lens[-1]}")
