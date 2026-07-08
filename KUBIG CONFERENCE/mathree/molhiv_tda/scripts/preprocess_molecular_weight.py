#!/usr/bin/env python3
"""Precompute and cache molecular weight features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import CACHE_ROOT, MW_CACHE, MW_STATS_CACHE
from data.load_molhiv import load_molhiv, normalize_train_stats
from features.molecular_weight import compute_molecular_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset, split_idx, _, smiles = load_molhiv(args.dataset_root)

    weights, failures = compute_molecular_weights(smiles)
    normalized, stats = normalize_train_stats(weights.squeeze(1), split_idx["train"])
    normalized = normalized.unsqueeze(1)

    torch.save(weights, MW_CACHE.with_name("molecular_weight_raw.pt"))
    torch.save({"features": normalized}, MW_CACHE)
    torch.save(stats, MW_STATS_CACHE)

    print(f"Saved molecular weights to {MW_CACHE}")
    print(f"Failures: {failures}/{len(smiles)}")


if __name__ == "__main__":
    main()
