#!/usr/bin/env python3
"""Diagnose Freebase class-wise mixing runs for GTN attention NaNs.

Reads only runs/class_wise_mixing/freebase__*/metrics.json and excludes smoke seed 0.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path("runs/class_wise_mixing")
SEEDS = [312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219]
FINAL_KEYS = ["test_macro_f1", "test_accuracy", "val_macro_f1"]
GTN_KEYS = ["gtn_only_test_macro_f1", "gtn_only_test_accuracy", "gtn_only_val_macro_f1"]


def has_nan(obj: Any) -> bool:
    if isinstance(obj, float):
        return math.isnan(obj)
    if isinstance(obj, dict):
        return any(has_nan(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_nan(v) for v in obj)
    return False


def is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if is_finite(x):
        return f"{float(x):.{digits}f}"
    return "-"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ROOT.glob("freebase__*/metrics.json")):
        name = path.parent.name
        if name.endswith("_s0"):
            continue
        parts = name.split("__")
        if len(parts) < 3:
            continue
        backbone = parts[1]
        try:
            seed = int(parts[2].rsplit("_s", 1)[1])
        except Exception:
            continue
        if seed not in SEEDS:
            continue
        with path.open() as f:
            m = json.load(f)
        mix = m.get("class_wise_mixing") or {}
        mixed = mix.get("mixed_nodes")
        unchanged = mix.get("unchanged_nodes")
        denom = (mixed or 0) + (unchanged or 0) if mixed is not None and unchanged is not None else 0
        mixed_ratio = float(mixed) / denom if denom else float("nan")
        rows.append({
            "backbone": backbone,
            "seed": seed,
            "path": str(path),
            "nan_gtn_attention": has_nan(m.get("gtn_attentions")),
            "nan_final_metric": any(has_nan(m.get(k)) for k in FINAL_KEYS),
            "mixed_nodes": mixed,
            "unchanged_nodes": unchanged,
            "mixed_ratio": mixed_ratio,
            **{k: m.get(k) for k in FINAL_KEYS + GTN_KEYS},
        })
    return sorted(rows, key=lambda r: (r["backbone"], SEEDS.index(r["seed"])))


def print_per_run(rows: list[dict[str, Any]]) -> None:
    cols = [
        "backbone", "seed", "nan_attn", "nan_final", "mixed_ratio",
        "gtn_f1", "gtn_acc", "gtn_val_f1", "final_f1", "final_acc", "final_val_f1", "path",
    ]
    print("\t".join(cols))
    for r in rows:
        vals = [
            r["backbone"],
            str(r["seed"]),
            "yes" if r["nan_gtn_attention"] else "no",
            "yes" if r["nan_final_metric"] else "no",
            fmt(r["mixed_ratio"]),
            fmt(r.get("gtn_only_test_macro_f1")),
            fmt(r.get("gtn_only_test_accuracy")),
            fmt(r.get("gtn_only_val_macro_f1")),
            fmt(r.get("test_macro_f1")),
            fmt(r.get("test_accuracy")),
            fmt(r.get("val_macro_f1")),
            r["path"],
        ]
        print("\t".join(vals))


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["backbone"]].append(r)
    print("\nsummary_by_backbone")
    print("backbone\ttotal\tnan_attention\tnan_final\tmean_mixed_ratio\tmean_final_f1\tmean_final_acc")
    for backbone in sorted(grouped):
        xs = grouped[backbone]
        ratios = [r["mixed_ratio"] for r in xs if is_finite(r["mixed_ratio"])]
        f1s = [r["test_macro_f1"] for r in xs if is_finite(r.get("test_macro_f1"))]
        accs = [r["test_accuracy"] for r in xs if is_finite(r.get("test_accuracy"))]
        print("\t".join([
            backbone,
            str(len(xs)),
            str(sum(1 for r in xs if r["nan_gtn_attention"])),
            str(sum(1 for r in xs if r["nan_final_metric"])),
            fmt(mean(ratios) if ratios else float("nan")),
            fmt(mean(f1s) if f1s else float("nan")),
            fmt(mean(accs) if accs else float("nan")),
        ]))


def main() -> None:
    rows = load_rows()
    print_per_run(rows)
    print_summary(rows)
    print("\nnotes")
    print("- NaN in gtn_attentions is a Stage-1 GTN diagnostic and can coincide with degenerate/NaN discovered adjacency H.")
    print("- Final finite metrics are reported separately; do not silently drop runs unless final metrics are NaN.")
    print("- Low Freebase mixed_ratio reflects small same-class same-split groups and many unchanged singleton nodes.")


if __name__ == "__main__":
    main()
