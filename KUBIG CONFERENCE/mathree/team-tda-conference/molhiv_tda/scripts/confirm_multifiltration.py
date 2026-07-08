#!/usr/bin/env python3
"""Confirm top filtration combos over multiple seeds; report mean/std."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

# top screen candidates + vanilla reference
COMBOS = ["", "C", "B,C", "A,B,C"]
SEEDS = [0, 1, 2]
LR = 1e-4
DROPOUT = 0.4


def tag(spec: str) -> str:
    return "vanilla" if not spec else "".join(s.strip().upper() for s in spec.split(","))


def run(spec: str, seed: int, epochs: int, device: str) -> dict:
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    out = RESULTS_ROOT / f"multifilt_{tag(spec)}{seed_suffix}.json"
    if not out.exists():
        cmd = [
            sys.executable, str(PROJECT_ROOT / "train" / "train_pdgnn_multifiltration.py"),
            "--filtrations", spec,
            "--lr", str(LR), "--dropout", str(DROPOUT),
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

    summary = []
    for spec in COMBOS:
        valids, tests = [], []
        for seed in SEEDS:
            r = run(spec, seed, args.epochs, args.device)
            valids.append(r["valid_rocauc"])
            tests.append(r["test_rocauc"])
        summary.append((tag(spec), np.mean(valids), np.std(valids),
                        np.mean(tests), np.std(tests), tests))

    summary.sort(key=lambda x: x[3], reverse=True)
    print("\n| Filtrations | Valid (mean±std) | Test (mean±std) | Test per seed |")
    print("| --- | --- | --- | --- |")
    for name, vm, vs, tm, ts, tests in summary:
        per = ", ".join(f"{t:.4f}" for t in tests)
        print(f"| {name} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | {per} |")


if __name__ == "__main__":
    main()
