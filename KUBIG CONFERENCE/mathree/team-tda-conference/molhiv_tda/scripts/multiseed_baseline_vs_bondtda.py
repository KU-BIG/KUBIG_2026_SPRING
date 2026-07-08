#!/usr/bin/env python3
"""Run PDGNN baseline vs. +BondTDA over multiple seeds and compare mean/std."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

SEEDS = [0, 1, 2]
LR = 1e-4
DROPOUT = 0.4


def run_baseline(seed: int, epochs: int, device: str) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    out = RESULTS_ROOT / f"pdgnn_lr{LR}_dropout{DROPOUT}{suffix}.json"
    if not out.exists():
        cmd = [
            sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn.py"),
            "--lr", str(LR), "--dropout", str(DROPOUT),
            "--epochs", str(epochs), "--device", device, "--seed", str(seed),
        ]
        print("Running", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out


def run_bond_tda(seed: int, epochs: int, device: str) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    out = RESULTS_ROOT / f"pdgnn_bond_tda{suffix}.json"
    if not out.exists():
        cmd = [
            sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn_tda.py"),
            "--config", "pdgnn_bond_tda",
            "--lr", str(LR), "--dropout", str(DROPOUT),
            "--epochs", str(epochs), "--device", device, "--seed", str(seed),
        ]
        print("Running", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out


def summarize(name: str, paths: list[Path]):
    valids, tests = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        valids.append(r["valid_rocauc"])
        tests.append(r["test_rocauc"])
    print(f"\n{name} (n={len(paths)})")
    print(f"  valid: {np.mean(valids):.4f} +/- {np.std(valids):.4f}  {['%.4f' % v for v in valids]}")
    print(f"  test:  {np.mean(tests):.4f} +/- {np.std(tests):.4f}  {['%.4f' % t for t in tests]}")
    return np.mean(valids), np.std(valids), np.mean(tests), np.std(tests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    baseline_paths = [run_baseline(s, args.epochs, args.device) for s in SEEDS]
    bond_tda_paths = [run_bond_tda(s, args.epochs, args.device) for s in SEEDS]

    print("\n" + "=" * 60)
    summarize("PDGNN baseline", baseline_paths)
    summarize("PDGNN + BondTDA", bond_tda_paths)


if __name__ == "__main__":
    main()
