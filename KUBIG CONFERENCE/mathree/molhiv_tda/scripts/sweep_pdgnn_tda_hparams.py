#!/usr/bin/env python3
"""Grid search over lr x dropout x weight_decay for a train_pdgnn_tda config.

Each combination is trained via train/train_pdgnn_tda.py and its result JSON is
written to results/sweep/<config>/ so runs never clobber each other. At the end
a ranked table (by valid ROC-AUC) is printed.

Example:
    python scripts/sweep_pdgnn_tda_hparams.py --config pdgnn_tda_3d_elec --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

# Default search space (override on the CLI with comma-separated values).
LRS = [1e-3, 5e-4, 1e-4]
DROPOUTS = [0.5, 0.3]
WEIGHT_DECAYS = [0.0, 1e-5]


def _floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def _tag(lr: float, dropout: float, wd: float) -> str:
    return f"lr{lr:g}_do{dropout:g}_wd{wd:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pdgnn_tda_3d_elec")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrs", type=_floats, default=LRS,
                        help="Comma-separated learning rates, e.g. 1e-3,5e-4,1e-4")
    parser.add_argument("--dropouts", type=_floats, default=DROPOUTS,
                        help="Comma-separated dropout rates, e.g. 0.5,0.3")
    parser.add_argument("--weight-decays", type=_floats, default=WEIGHT_DECAYS,
                        help="Comma-separated weight decays, e.g. 0.0,1e-5")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even if a result JSON already exists.")
    args = parser.parse_args()

    sweep_dir = RESULTS_ROOT / "sweep" / args.config
    sweep_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.lrs, args.dropouts, args.weight_decays))
    print(f"Sweeping {len(grid)} combos for config={args.config} "
          f"(epochs={args.epochs}, seed={args.seed})")

    rows = []
    for i, (lr, dropout, wd) in enumerate(grid, 1):
        out = sweep_dir / f"{_tag(lr, dropout, wd)}.json"
        if out.exists() and not args.rerun:
            print(f"[{i}/{len(grid)}] Skipping {out.name} (exists)")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn_tda.py"),
                "--config", args.config,
                "--lr", str(lr),
                "--dropout", str(dropout),
                "--weight-decay", str(wd),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(args.seed),
                "--out", str(out),
            ]
            print(f"[{i}/{len(grid)}] Running {_tag(lr, dropout, wd)}")
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: r.get("valid_rocauc", -1), reverse=True)
    print("\n| LR | Dropout | Weight decay | Valid ROC-AUC | Test ROC-AUC |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r.get('lr')} | {r.get('dropout')} | {r.get('weight_decay')} "
              f"| {r.get('valid_rocauc', float('nan')):.4f} "
              f"| {r.get('test_rocauc', float('nan')):.4f} |")

    if rows:
        best = rows[0]
        print(f"\nBest: lr={best.get('lr')} dropout={best.get('dropout')} "
              f"weight_decay={best.get('weight_decay')} "
              f"-> valid={best.get('valid_rocauc'):.4f} test={best.get('test_rocauc'):.4f}")
        summary = sweep_dir / "_summary.json"
        with open(summary, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Summary saved to {summary}")


if __name__ == "__main__":
    main()
