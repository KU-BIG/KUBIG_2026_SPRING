#!/usr/bin/env python3
"""Run HAN+PDGNN fusion ablation experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ALL_MODELS = ["gcn", "han_only", "pdgnn_only", "concat", "main"]


def main():
    parser = argparse.ArgumentParser(description="Run HAN+PDGNN fusion ablations")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="main", choices=ALL_MODELS + ["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    for name in models:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train_han_pdgnn.py"),
            "--config",
            args.config,
            "--model",
            name,
            "--seed",
            str(args.seed),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.device is not None:
            cmd += ["--device", args.device]
        print("Running:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    if args.model == "all" and not args.dry_run:
        print("\n| Model | Valid ROC-AUC | Test ROC-AUC |")
        print("| --- | ---: | ---: |")
        for name in ALL_MODELS:
            path = PROJECT_ROOT / "results" / f"han_pdgnn_{name}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    r = json.load(f)
                print(f"| {r.get('model', name)} | {r['valid_rocauc']:.4f} | {r['test_rocauc']:.4f} |")


if __name__ == "__main__":
    main()
