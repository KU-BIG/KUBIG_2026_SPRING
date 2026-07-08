#!/usr/bin/env python3
"""Precompute per-edge [distance, coulomb_like] features using RDKit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CACHE_ROOT, EDGE_ELECTRO_CACHE
from data.load_molhiv import load_molhiv
from features.conformer_3d import compute_edge_electrostatic_for_graph


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

    edge_phys_bank: list[torch.Tensor] = [torch.zeros((0, 2), dtype=torch.float32) for _ in range(len(dataset))]
    failures = 0
    for idx in tqdm(indices, desc="edge electrostatic"):
        data = dataset[idx]
        feat, ok = compute_edge_electrostatic_for_graph(
            smiles[idx], data.edge_index, int(data.num_nodes)
        )
        edge_phys_bank[idx] = torch.tensor(feat, dtype=torch.float32)
        if not ok:
            failures += 1

    torch.save({"edge_phys": edge_phys_bank, "failures": failures}, EDGE_ELECTRO_CACHE)
    print(f"Saved edge electrostatic features to {EDGE_ELECTRO_CACHE}")
    print(f"Failures: {failures}/{len(indices)}")


if __name__ == "__main__":
    main()
