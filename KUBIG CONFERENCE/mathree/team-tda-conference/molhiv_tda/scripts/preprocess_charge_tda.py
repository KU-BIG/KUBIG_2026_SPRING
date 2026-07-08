#!/usr/bin/env python3
"""Precompute Gasteiger partial-charge filtration TDA features (lens D).

Includes a sanity check that the persistence images are not near-zero (the
failure mode that silently killed the earlier bond-TDA experiments).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import CACHE_ROOT, CHARGE_TDA_CACHE
from data.load_molhiv import load_molhiv, normalize_train_stats
from features.charge_filtration_tda import compute_charge_tda_batch


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

    features, failures = compute_charge_tda_batch(dataset, smiles, indices=indices)

    # Sanity check BEFORE normalization: fraction of exactly-zero cells and mean
    # magnitude. If the image is ~all zero the sigma/range is mis-scaled.
    nonzero_frac = float((features != 0).float().mean())
    print(
        f"[D] raw persistence-image stats: mean={features.mean():.4f} "
        f"max={features.max():.4f} nonzero_frac={nonzero_frac:.3f}"
    )
    if nonzero_frac < 0.05:
        print(
            "WARNING: persistence images are ~all zero -- charge sigma/range is "
            "likely mis-scaled. Inspect before trusting downstream results."
        )

    full = torch.zeros((len(dataset), features.shape[1]), dtype=torch.float32)
    full[indices] = features
    normalized, _ = normalize_train_stats(full, split_idx["train"])
    torch.save({"features": normalized}, CHARGE_TDA_CACHE)
    print(f"[D] saved {tuple(normalized.shape)} -> {CHARGE_TDA_CACHE}")
    print(f"[D] SMILES parse failures: {failures}/{len(indices)}")


if __name__ == "__main__":
    main()
