#!/usr/bin/env python3
"""Dropout sweep for the A+B+C multi-filtration model (seeds 0,1,2)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

FILTRATIONS = "A,B,C"
COMBO_TAG = "ABC"
DROPOUTS = [0.4, 0.5, 0.6, 0.7]  # 0.4 already has seed 0,1,2 from confirmation
SEEDS = [0, 1, 2]
LR = 1e-4
DEFAULT_DROPOUT = 0.4  # matches config.DROPOUT -> no suffix in filename


def result_path(dropout: float, seed: int) -> Path:
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    drop_suffix = "" if dropout == DEFAULT_DROPOUT else f"_d{dropout}"
    return RESULTS_ROOT / f"multifilt_{COMBO_TAG}{seed_suffix}{drop_suffix}.json"


def run(dropout: float, seed: int, epochs: int, device: str) -> dict:
    out = result_path(dropout, seed)
    if not out.exists():
        cmd = [
            sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn_multifiltration.py"),
            "--filtrations", FILTRATIONS,
            "--lr", str(LR), "--dropout", str(dropout),
            "--epochs", str(epochs), "--device", device, "--seed", str(seed),
        ]
        print("Running", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rows = []
    for dropout in DROPOUTS:
        valids, tests = [], []
        for seed in SEEDS:
            r = run(dropout, seed, args.epochs, args.device)
            valids.append(r["valid_rocauc"])
            tests.append(r["test_rocauc"])
        rows.append((dropout, np.mean(valids), np.std(valids),
                     np.mean(tests), np.std(tests), tests))

    rows.sort(key=lambda x: x[3], reverse=True)
    print("\n| Dropout | Valid (mean±std) | Test (mean±std) | Test per seed |")
    print("| ---: | --- | --- | --- |")
    for d, vm, vs, tm, ts, tests in rows:
        per = ", ".join(f"{t:.4f}" for t in tests)
        print(f"| {d} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | {per} |")


if __name__ == "__main__":
    main()
