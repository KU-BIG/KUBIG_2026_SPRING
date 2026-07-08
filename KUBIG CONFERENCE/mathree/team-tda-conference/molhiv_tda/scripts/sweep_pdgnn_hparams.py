#!/usr/bin/env python3
"""Small lr x dropout grid search for the PDGNN baseline."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

LRS = [1e-3, 1e-4, 1e-5]
DROPOUTS = [0.5, 0.4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    args = parser.parse_args()

    suffix = "" if args.loss == "bce" else f"_{args.loss}"
    rows = []
    for lr, dropout in itertools.product(LRS, DROPOUTS):
        out = RESULTS_ROOT / f"pdgnn_lr{lr}_dropout{dropout}{suffix}.json"
        if out.exists():
            print(f"Skipping lr={lr} dropout={dropout} (found {out})")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn.py"),
                "--lr", str(lr),
                "--dropout", str(dropout),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--loss", args.loss,
            ]
            print("Running", " ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: r["valid_rocauc"], reverse=True)
    print("\n| LR | Dropout | Valid ROC-AUC | Test ROC-AUC |")
    print("| ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['lr']} | {r['dropout']} | {r['valid_rocauc']:.4f} | {r['test_rocauc']:.4f} |")


if __name__ == "__main__":
    main()
