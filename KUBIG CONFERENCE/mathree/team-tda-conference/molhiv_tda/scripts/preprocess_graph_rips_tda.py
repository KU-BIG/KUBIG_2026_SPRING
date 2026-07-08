#!/usr/bin/env python3
"""Precompute hop-distance Rips TDA features (lenses A and B)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    AROMATIC_RIPS_TDA_CACHE,
    CACHE_ROOT,
    GRAPH_RIPS_TDA_CACHE,
)
from data.load_molhiv import load_molhiv, normalize_train_stats
from features.graph_rips_tda import compute_graph_rips_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--lens", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset, split_idx, _, _ = load_molhiv(args.dataset_root)

    indices = list(range(len(dataset)))
    if args.max_graphs is not None:
        indices = indices[: args.max_graphs]

    targets = []
    if args.lens in ("A", "both"):
        targets.append(("A", False, GRAPH_RIPS_TDA_CACHE))
    if args.lens in ("B", "both"):
        targets.append(("B", True, AROMATIC_RIPS_TDA_CACHE))

    for name, aromatic_only, cache in targets:
        features = compute_graph_rips_batch(dataset, aromatic_only=aromatic_only, indices=indices)
        full = torch.zeros((len(dataset), features.shape[1]), dtype=torch.float32)
        full[indices] = features
        normalized, _ = normalize_train_stats(full, split_idx["train"])
        torch.save({"features": normalized}, cache)
        print(f"[{name}] saved {tuple(normalized.shape)} -> {cache}")


if __name__ == "__main__":
    main()
