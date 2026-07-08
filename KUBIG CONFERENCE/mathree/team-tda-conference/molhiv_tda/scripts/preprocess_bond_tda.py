#!/usr/bin/env python3
"""Precompute and cache bond-type TDA features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import BOND_TDA_CACHE, CACHE_ROOT, TDA_STATS_CACHE
from data.load_molhiv import load_molhiv, normalize_train_stats
from features.bond_filtration_tda import compute_bond_tda_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset, split_idx, _, _ = load_molhiv(args.dataset_root)

    indices = list(range(len(dataset)))
    if args.max_graphs is not None:
        indices = indices[: args.max_graphs]

    features = compute_bond_tda_batch(dataset, indices=indices)
    full = torch.zeros((len(dataset), features.shape[1]), dtype=torch.float32)
    full[indices] = features

    normalized, stats = normalize_train_stats(full, split_idx["train"])
    torch.save({"features": normalized}, BOND_TDA_CACHE)
    torch.save(stats, TDA_STATS_CACHE)

    print(f"Saved bond-type TDA features to {BOND_TDA_CACHE}")
    print(f"Feature shape: {tuple(normalized.shape)}")


if __name__ == "__main__":
    main()
