#!/usr/bin/env python3
"""Precompute and cache 3D conformer distance-based TDA features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from config import CACHE_ROOT, TDA_3D_CACHE, TDA_3D_STATS_CACHE, TDA_3D_DIM
from data.load_molhiv import load_molhiv, normalize_train_stats
from features.conformer_3d import compute_3d_tda_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset, split_idx, _, smiles = load_molhiv(args.dataset_root)

    indices = list(range(len(dataset)))
    if args.max_graphs is not None:
        indices = indices[: args.max_graphs]

    features = np.zeros((len(dataset), TDA_3D_DIM), dtype=np.float32)
    failures = 0

    for idx in tqdm(indices, desc="3D TDA"):
        vec, ok = compute_3d_tda_vector(smiles[idx])
        features[idx] = vec
        if not ok:
            failures += 1

    feat_tensor = torch.tensor(features, dtype=torch.float32)
    normalized, stats = normalize_train_stats(feat_tensor, split_idx["train"])
    torch.save({"features": normalized, "failures": failures}, TDA_3D_CACHE)
    torch.save(stats, TDA_3D_STATS_CACHE)

    print(f"Saved 3D TDA features to {TDA_3D_CACHE}")
    print(f"Failures: {failures}/{len(indices)}")


if __name__ == "__main__":
    main()
