#!/usr/bin/env bash
# What can pier do besides `run`? Grading a patch we supply, at an intermediate
# point of a recorded trajectory, is the whole of the intermediate-label plan,
# and nothing in our own scripts does it. Before designing around a guess, ask.
set -uo pipefail
echo "=== deps ==="
python -m pip install -q --no-cache-dir "datacurve-pier==0.3.0" 2>&1 | tail -2
python -c "import pier; print('[probe] pier', getattr(pier, '__version__', 'ok'))" || exit 21

echo; echo "=== top-level CLI ==="
pier --help 2>&1 | head -40

echo; echo "=== subcommands ==="
for sub in run eval verify grade score test judge submit; do
  if pier "$sub" --help >/tmp/h 2>&1; then
    echo "--- pier $sub ---"; head -22 /tmp/h
  fi
done

echo; echo "=== python surface ==="
python - <<'PY'
import inspect, pkgutil
import pier
print("pier dir:", [n for n in dir(pier) if not n.startswith("_")][:40])
for m in pkgutil.iter_modules(pier.__path__):
    print("  module:", m.name)
for name in ("verify", "verifier", "grade", "grading", "harness", "runner", "task"):
    try:
        mod = __import__(f"pier.{name}", fromlist=["x"])
    except Exception:
        continue
    fns = [n for n, o in vars(mod).items() if not n.startswith("_") and callable(o)]
    print(f"  pier.{name}: {fns[:18]}")
PY
echo "[probe] done"
