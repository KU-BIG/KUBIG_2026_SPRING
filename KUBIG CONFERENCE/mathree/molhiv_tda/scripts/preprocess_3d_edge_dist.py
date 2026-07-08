#!/usr/bin/env python3
"""Precompute 3D conformer edge distances for PDGNN distance filtration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from config import CACHE_ROOT, EDGE_DIST_CACHE
from data.load_molhiv import load_molhiv
from features.conformer_3d import compute_edge_distances_for_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset, _, _, smiles = load_molhiv(args.dataset_root)

    indices = list(range(len(dataset)))
    if args.max_graphs is not None:
        indices = indices[: args.max_graphs]

    dist_bank: list[torch.Tensor] = [torch.zeros(0)] * len(dataset)
    failures = 0

    for idx in tqdm(indices, desc="3D edge distances"):
        data = dataset[idx]
        dists, ok = compute_edge_distances_for_graph(
            smiles[idx], data.edge_index, int(data.num_nodes)
        )
        dist_bank[idx] = torch.tensor(dists, dtype=torch.float32)
        if not ok:
            failures += 1

    torch.save({"distances": dist_bank, "failures": failures}, EDGE_DIST_CACHE)
    print(f"Saved edge distances to {EDGE_DIST_CACHE}")
    print(f"Failures: {failures}/{len(indices)}")


if __name__ == "__main__":
    main()
