#!/usr/bin/env python3
"""Summarize class-wise mixing topology ablation results.

Reads only runs/class_wise_mixing/*/metrics.json, excludes smoke seed 0, and writes
results/CLASS_WISE_MIXING.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
BACKBONES = ["han", "rgcn"]
SEEDS = [312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219]
ROOT = Path("runs/class_wise_mixing")
OUT = Path("results/CLASS_WISE_MIXING.md")


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def has_nan(x: Any) -> bool:
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, dict):
        return any(has_nan(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(has_nan(v) for v in x)
    return False


def fmt_num(x: Any, digits: int = 4) -> str:
    if not is_finite_number(x):
        return "nan" if isinstance(x, float) and math.isnan(x) else "-"
    return f"{float(x):.{digits}f}"


def fmt_mean_std(vals: list[float], digits: int = 4) -> str:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return "-"
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean(vals):.{digits}f}+/-{sd:.{digits}f}"


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for seed in SEEDS:
                path = ROOT / f"{ds}__{bb}__class_wise_mixing_s{seed}" / "metrics.json"
                if not path.exists():
                    missing.append({"dataset": ds, "backbone": bb, "seed": seed, "path": str(path)})
                    continue
                with path.open() as f:
                    m = json.load(f)
                mix = m.get("class_wise_mixing") or {}
                mixed = mix.get("mixed_nodes")
                unchanged = mix.get("unchanged_nodes")
                denom = (mixed or 0) + (unchanged or 0) if mixed is not None and unchanged is not None else 0
                ratio = float(mixed) / denom if denom else None
                final_finite = all(is_finite_number(m.get(k)) for k in ("test_macro_f1", "test_accuracy", "val_macro_f1"))
                rows.append({
                    "dataset": ds,
                    "backbone": bb,
                    "seed": seed,
                    "path": str(path),
                    "test_macro_f1": m.get("test_macro_f1"),
                    "test_accuracy": m.get("test_accuracy"),
                    "val_macro_f1": m.get("val_macro_f1"),
                    "gtn_only_test_macro_f1": m.get("gtn_only_test_macro_f1"),
                    "gtn_only_test_accuracy": m.get("gtn_only_test_accuracy"),
                    "gtn_only_val_macro_f1": m.get("gtn_only_val_macro_f1"),
                    "mixed_nodes": mixed,
                    "unchanged_nodes": unchanged,
                    "mixed_ratio": ratio,
                    "nan_gtn_attention": has_nan(m.get("gtn_attentions")),
                    "final_metrics_finite": final_finite,
                })
    return rows, missing


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["backbone"])].append(r)
    return grouped


def summary_table(rows: list[dict[str, Any]]) -> list[str]:
    grouped = group_rows(rows)
    lines = [
        "| dataset | backbone | n | finite n | test_macro_f1 mean+/-std | test_accuracy mean+/-std | val_macro_f1 mean+/-std | mixed ratio mean+/-std | NaN GTN attention runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for bb in BACKBONES:
            xs = sorted(grouped.get((ds, bb), []), key=lambda r: SEEDS.index(r["seed"]))
            if not xs:
                continue
            finite = [r for r in xs if r["final_metrics_finite"]]
            ratios = [r["mixed_ratio"] for r in xs if is_finite_number(r.get("mixed_ratio"))]
            lines.append("| " + " | ".join([
                ds,
                bb.upper(),
                str(len(xs)),
                str(len(finite)),
                fmt_mean_std([r["test_macro_f1"] for r in finite]),
                fmt_mean_std([r["test_accuracy"] for r in finite]),
                fmt_mean_std([r["val_macro_f1"] for r in finite]),
                fmt_mean_std(ratios),
                str(sum(1 for r in xs if r["nan_gtn_attention"])),
            ]) + " |")
    return lines


def pair_status(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    grouped = group_rows(rows)
    completed = []
    incomplete = []
    for ds in DATASETS:
        for bb in BACKBONES:
            n = len(grouped.get((ds, bb), []))
            line = f"- {ds} / {bb.upper()}: n={n}/10"
            (completed if n == 10 else incomplete).append(line)
    return completed, incomplete


def per_run_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| dataset | backbone | seed | test_macro_f1 | test_accuracy | val_macro_f1 | mixed_nodes | unchanged_nodes | mixed_ratio | gtn_only_test_macro_f1 | gtn_only_test_accuracy | nan_gtn_attention | final_metrics_finite | metrics path |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    order = {(ds, bb, seed): (DATASETS.index(ds), BACKBONES.index(bb), SEEDS.index(seed))
             for ds in DATASETS for bb in BACKBONES for seed in SEEDS}
    for r in sorted(rows, key=lambda x: order[(x["dataset"], x["backbone"], x["seed"])]):
        lines.append("| " + " | ".join([
            r["dataset"],
            r["backbone"].upper(),
            str(r["seed"]),
            fmt_num(r["test_macro_f1"]),
            fmt_num(r["test_accuracy"]),
            fmt_num(r["val_macro_f1"]),
            str(r["mixed_nodes"] if r["mixed_nodes"] is not None else "-"),
            str(r["unchanged_nodes"] if r["unchanged_nodes"] is not None else "-"),
            fmt_num(r["mixed_ratio"]),
            fmt_num(r["gtn_only_test_macro_f1"]),
            fmt_num(r["gtn_only_test_accuracy"]),
            "yes" if r["nan_gtn_attention"] else "no",
            "yes" if r["final_metrics_finite"] else "no",
            f"`{r['path']}`",
        ]) + " |")
    return lines


def missing_table(missing: list[dict[str, Any]]) -> list[str]:
    if not missing:
        return ["No missing official runs."]
    lines = ["| dataset | backbone | seed | expected metrics path |", "|---|---|---:|---|"]
    for m in missing:
        lines.append(f"| {m['dataset']} | {m['backbone'].upper()} | {m['seed']} | `{m['path']}` |")
    return lines


def diagnostics(rows: list[dict[str, Any]]) -> list[str]:
    freebase = [r for r in rows if r["dataset"] == "freebase"]
    fb_nan = sum(1 for r in freebase if r["nan_gtn_attention"])
    fb_finite = sum(1 for r in freebase if r["final_metrics_finite"])
    fb_ratios = [r["mixed_ratio"] for r in freebase if is_finite_number(r.get("mixed_ratio"))]
    lines = [
        "- Freebase GTN attention diagnostic: "
        f"{fb_nan}/{len(freebase)} completed Freebase runs contain NaN values in `gtn_attentions`; "
        f"{fb_finite}/{len(freebase)} have finite final `test_macro_f1`, `test_accuracy`, and `val_macro_f1`.",
    ]
    if fb_ratios:
        lines.append(f"- Freebase mixed ratio: {fmt_mean_std(fb_ratios)} across completed Freebase runs.")
    lines += [
        "- NaN values in saved GTN attentions suggest GTN-stage instability or degenerate attention on Freebase. These runs are retained when final metrics are finite, but the attention diagnostics should not be hidden.",
        "- Class-wise mixing can be weak when same-class same-split groups are small; singleton groups remain unchanged and are reported through `unchanged_nodes` and mixed ratio.",
        "- MAG has many classes and uses subsampled nodes, so macro-F1 can be very low. Interpret MAG primarily through within-dataset deltas rather than absolute macro-F1.",
    ]
    return lines


def build_markdown(rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    completed, incomplete = pair_status(rows)
    total = len(rows)
    finite_total = sum(1 for r in rows if r["final_metrics_finite"])
    status = "complete" if total == 140 else "in progress"
    md: list[str] = []
    md += [
        "# Class-wise Mixing Topology Ablation",
        "",
        "## Status",
        "",
        f"Generated: {generated}",
        "",
        f"Status: **{status}**. Completed official runs: **{total}/140**; finite final metrics: **{finite_total}/{total}**.",
        "",
        "Smoke-test seed `0` is excluded from all summaries. Official seeds: " + ", ".join(str(s) for s in SEEDS) + ".",
        "",
        "Completed dataset/backbone pairs:",
        "",
        *(completed if completed else ["- None"]),
        "",
        "Incomplete dataset/backbone pairs:",
        "",
        *(incomplete if incomplete else ["- None"]),
        "",
        "## Motivation",
        "",
        "Class-wise mixing is a structured topology control inspired by CFH-style within-class feature mixing. The run first computes the real GTN-PDGNN topology feature, then before concatenation replaces each node's topology feature with another node's topology feature from the same class, preferably within the same train/val/test split.",
        "",
        "Random/noisy feature controls test whether simply adding feature dimensions helps. Class-wise mixing is a stronger counterfactual for topology features: it preserves class-level topology-feature distribution to some extent while breaking node-specific topology-feature alignment. If real GTN-PDGNN topology outperforms class-wise mixed topology, that supports the claim that the topology feature carries meaningful node-specific signal.",
        "",
        "## Experimental Setup",
        "",
        "- Datasets: `acm`, `dblp`, `imdb`, `freebase`, `mag`, `aifb`, `yelp`.",
        "- Backbones: HAN and RGCN.",
        "- Topology source: `gtn`.",
        "- Topology mode: `class_wise_mixing`.",
        "- Expected total: 7 datasets x 2 backbones x 10 seeds = 140 runs.",
        "- Result path pattern: `runs/class_wise_mixing/<dataset>__<backbone>__class_wise_mixing_s<seed>/metrics.json`.",
        "- The topology cache stores real GTN-PDGNN topology features before class-wise mixing. It is computation reuse only and does not change the model or ablation definition. Class-wise mixing is applied after loading or computing the real topology feature.",
        "",
        "## Current Results Table",
        "",
        *summary_table(rows),
        "",
        "## Missing Official Runs",
        "",
        *missing_table(missing),
        "",
        "## Per-run Table",
        "",
        *per_run_table(rows),
        "",
        "## Diagnostics / Caveats",
        "",
        *diagnostics(rows),
        "- If this document reports fewer than 140 completed runs, the table is interim and should be regenerated after resume completes.",
        "",
        "## Reproduction Commands",
        "",
        "Resume missing runs:",
        "",
        "```bash",
        "source /opt/miniforge3/etc/profile.d/conda.sh",
        "conda activate tda",
        "cd /workspace/TDA",
        "bash experiments/run_class_wise_mixing.sh",
        "```",
        "",
        "Count completed official runs:",
        "",
        "```bash",
        "find runs/class_wise_mixing -name metrics.json | grep -Ev '_s0/metrics.json$' | wc -l",
        "```",
        "",
        "Check missing official runs:",
        "",
        "```bash",
        "python - <<'PY'",
        "from pathlib import Path",
        "datasets=['acm','dblp','imdb','freebase','mag','aifb','yelp']",
        "backbones=['han','rgcn']",
        "seeds=[312132,238623,792965,15092,661491,588722,825661,500973,88015,251219]",
        "root=Path('runs/class_wise_mixing')",
        "missing=[]",
        "for ds in datasets:",
        "  for bb in backbones:",
        "    for s in seeds:",
        "      p=root/f'{ds}__{bb}__class_wise_mixing_s{s}'/'metrics.json'",
        "      if not p.exists(): missing.append(str(p))",
        "print(f'completed={140-len(missing)} missing={len(missing)} expected=140')",
        "print('\\n'.join(missing))",
        "PY",
        "```",
        "",
        "Regenerate this summary:",
        "",
        "```bash",
        "python experiments/summarize_class_wise_mixing.py",
        "```",
        "",
    ]
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    rows, missing = load_rows()
    text = build_markdown(rows, missing)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("\n".join(summary_table(rows)))
    print(f"\nwrote {out} ({len(rows)}/140 official runs, missing {len(missing)})")


if __name__ == "__main__":
    main()
