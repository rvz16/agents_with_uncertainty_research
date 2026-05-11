"""For a fixed (benchmark, generator) cell, render N bar charts — one per
c_ver value in the sweep — so the audience can see the policy ordering
change as verification cost grows.

Output: data/paper_figs/fig_cver_bars_<bench>_<gen>_cver<value>.png

Usage:
  python3 scripts/fig_cver_bars.py --data-root data --out-dir data/paper_figs \
    --benchmark lcb_calibration_v2 --generator gpt5_mini \
    --cvers 10,20,40,60,100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICIES = [
    ("bayesian_greedy",  "bayes_greedy",  "#1d4ed8"),
    ("bayesian_DP",      "bayes_DP",      "#0ea5e9"),
    ("threshold_L0",     "threshold(L0)", "#94a3b8"),
    ("threshold_L2",     "threshold(L2)", "#22c55e"),
    ("threshold_L3",     "threshold(L3)", "#a855f7"),
    ("best_of_3",        "best_of_3",     "#f59e0b"),
    ("fixed_pipeline",   "fixed_pipeline","#ef4444"),
    ("always_verify",    "always_verify", "#525252"),
]


def render_one(d: dict, value: int, bench: str, gen: str, out_dir: Path,
               mode: str = "cver") -> Path:
    """mode: 'cver' (sweep c_ver, R=100) or 'reward' (sweep R, c_ver=30)."""
    if mode == "cver":
        cond = d["D2_c_ver_sweep"][f"c_ver_{value}"]
        title_param = f"$c_\\mathrm{{ver}}={value},\\ R=100$"
        slug_kind = "cver"
    elif mode == "reward":
        cond = d["D4_reward_sweep"][f"reward_{value}"]
        title_param = f"$R={value},\\ c_\\mathrm{{ver}}=30$"
        slug_kind = "R"
    else:
        raise ValueError(f"unknown mode: {mode}")
    keys = [k for k, _, _ in POLICIES if k in cond]
    deltas = [cond[k]["mean_utility"] for k in keys]
    labels = [next(lbl for k, lbl, _ in POLICIES if k == kk) for kk in keys]
    colors = [next(c for k, _, c in POLICIES if k == kk) for kk in keys]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(keys), dtype=float)
    bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.55)
    winner_idx = int(np.argmax(deltas))
    for i, (b, v) in enumerate(zip(bars, deltas)):
        offset = max(abs(v) * 0.05, 0.5)
        text = f"{'+' if v >= 0 else ''}{v:.1f}"
        if i == winner_idx:
            text = r"$\bigstar$ " + text
        ax.text(b.get_x() + b.get_width() / 2,
                v + offset if v >= 0 else v - offset,
                text,
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.0,
                fontweight="bold" if i == winner_idx else "normal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("mean utility", fontsize=9)
    ax.set_title(f"{bench} / {gen}  ({title_param})",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    bench_slug = bench.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    # Strip display-label suffix like " (API)" / " (local)" so the filename
    # stays compatible with existing slide \includegraphics references.
    gen_slug = gen.split(" ")[0]
    out_png = out_dir / f"fig_{slug_kind}_bars_{bench_slug}_{gen_slug}_{slug_kind}{value}.png"
    out_pdf = out_dir / f"fig_{slug_kind}_bars_{bench_slug}_{gen_slug}_{slug_kind}{value}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir",   required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=str,
                        help="benchmark dir name under data/, e.g. lcb_calibration_v2")
    parser.add_argument("--generator", required=True, type=str)
    parser.add_argument("--bench-label", default=None,
                        help="display label for the benchmark (default: derived)")
    parser.add_argument("--mode", choices=["cver", "reward"], default="cver",
                        help="which sweep to render: cver (default) or reward")
    parser.add_argument("--values", required=True, type=str,
                        help="comma-separated values to render (c_ver values if mode=cver, R values if mode=reward)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sens_path = args.data_root / args.benchmark / args.generator / "sensitivity.json"
    if not sens_path.exists():
        raise SystemExit(f"sensitivity.json not found at {sens_path}")
    d = json.loads(sens_path.read_text())

    bench_label = args.bench_label or args.benchmark.replace("lcb_calibration_v2", "LCB-hard").replace("_", " ")
    gen_label = {"gpt5_mini":"gpt-5-mini (API)","qwen3_coder":"qwen3-coder (API)",
                 "haiku45":"haiku-4.5 (API)","sonnet45":"sonnet-4.5 (API)",
                 "qwen25_32b":"qwen2.5-32b (local)"}.get(args.generator, args.generator)

    sweep_dict = d.get("D2_c_ver_sweep" if args.mode == "cver" else "D4_reward_sweep", {})
    sweep_key_prefix = "c_ver_" if args.mode == "cver" else "reward_"
    for v in [int(x) for x in args.values.split(",")]:
        if f"{sweep_key_prefix}{v}" not in sweep_dict:
            print(f"  {args.mode}={v} not in sweep, skipping")
            continue
        out = render_one(d, v, bench_label, gen_label, args.out_dir, mode=args.mode)
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
