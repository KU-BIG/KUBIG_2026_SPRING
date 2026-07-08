#!/usr/bin/env python3
"""Screen the 8 filtration combinations (single seed) and print a ranked table."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

# vanilla + 3 singles + 3 pairs + 1 triple
COMBOS = ["", "A", "B", "C", "A,B", "B,C", "C,A", "A,B,C"]
LR = 1e-4
DROPOUT = 0.4


def combo_tag(spec: str) -> str:
    return "vanilla" if not spec else "".join(s.strip().upper() for s in spec.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    rows = []
    for spec in COMBOS:
        tag = combo_tag(spec)
        out = RESULTS_ROOT / f"multifilt_{tag}{seed_suffix}.json"
        if not out.exists():
            cmd = [
                sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn_multifiltration.py"),
                "--filtrations", spec,
                "--lr", str(LR), "--dropout", str(DROPOUT),
                "--epochs", str(args.epochs), "--device", args.device, "--seed", str(args.seed),
            ]
            print("Running", " ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: r["test_rocauc"], reverse=True)
    print("\n| Filtrations | Valid AUC | Test AUC |")
    print("| --- | ---: | ---: |")
    for r in rows:
        tag = "+".join(r["filtrations"]) if r["filtrations"] else "vanilla"
        print(f"| {tag} | {r['valid_rocauc']:.4f} | {r['test_rocauc']:.4f} |")


if __name__ == "__main__":
    main()
