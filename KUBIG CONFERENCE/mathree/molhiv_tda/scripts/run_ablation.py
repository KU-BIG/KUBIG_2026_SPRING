#!/usr/bin/env python3
"""Run all PDGNN ablation experiments and print result table."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


EXPERIMENTS = [
    ("train/train_pdgnn.py", []),
    ("train/train_pdgnn_tda.py", ["--config", "pdgnn_mw"]),
    ("train/train_pdgnn_tda.py", ["--config", "pdgnn_bond_tda"]),
    ("train/train_pdgnn_tda.py", ["--config", "pdgnn_bond_tda_mw"]),
    ("train/train_pdgnn_tda.py", ["--config", "pdgnn_3d_tda"]),
    ("train/train_pdgnn_tda.py", ["--config", "pdgnn_bond_tda_3d_mw"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_preprocess:
        for script in (
            "scripts/preprocess_molecular_weight.py",
            "scripts/preprocess_bond_tda.py",
            "scripts/preprocess_3d_tda.py",
        ):
            cmd = [sys.executable, str(PROJECT_ROOT / script)]
            print("Running", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    for script, extra in EXPERIMENTS:
        cmd = [sys.executable, str(PROJECT_ROOT / script)] + extra
        cmd += ["--epochs", str(args.epochs), "--device", args.device]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        print("Running", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    print("\n| Model | Bond-type TDA | Molecular weight | 3D TDA | Valid ROC-AUC | Test ROC-AUC |")
    print("| --- | --- | --- | --- | ---: | ---: |")
    for path in sorted(RESULTS_ROOT.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        valid = r.get("valid_rocauc")
        test = r.get("test_rocauc")
        if isinstance(valid, (int, float)) and isinstance(test, (int, float)):
            print(
                f"| {r.get('model', path.stem)} | "
                f"{'Yes' if r.get('bond_type_tda') else 'No'} | "
                f"{'Yes' if r.get('molecular_weight') else 'No'} | "
                f"{'Yes' if r.get('tda_3d') else 'No'} | "
                f"{valid:.4f} | {test:.4f} |"
            )


if __name__ == "__main__":
    main()
