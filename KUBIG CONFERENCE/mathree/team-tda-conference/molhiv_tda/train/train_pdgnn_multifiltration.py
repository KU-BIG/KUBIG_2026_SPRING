#!/usr/bin/env python3
"""Train PDGNN with a concatenation of selected topological filtration lenses.

Lenses:
  A = full-graph hop-distance Rips        (config.GRAPH_RIPS_TDA_CACHE)
  B = aromatic-subgraph hop-distance Rips (config.AROMATIC_RIPS_TDA_CACHE)
  C = bond-type filtration                (config.BOND_TDA_CACHE, existing)
  D = Gasteiger partial-charge filtration (config.CHARGE_TDA_CACHE)

Each lens is a 50-dim persistence-image vector; selected lenses are concatenated
and fed to PDGNNTDA's head (simple concat fusion). Empty selection == vanilla PDGNN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    AROMATIC_RIPS_TDA_CACHE,
    BOND_TDA_CACHE,
    CHARGE_TDA_CACHE,
    CHARGE_TDA_DIM,
    DEFAULT_DEVICE,
    DROPOUT,
    EMB_DIM,
    EPOCHS,
    GRAPH_RIPS_DIM,
    GRAPH_RIPS_TDA_CACHE,
    BOND_TDA_DIM,
    NUM_BACKBONE_LAYERS,
    PATIENCE,
    RESULTS_ROOT,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import load_feature_tensor, load_molhiv, make_loaders
from models.pdgnn_baseline import PDGNNBaseline
from models.pdgnn_tda import PDGNNTDA
from train.train_utils import run_training, save_result
from utils.device import device_label, resolve_device

LENS_CACHE = {
    "A": (GRAPH_RIPS_TDA_CACHE, GRAPH_RIPS_DIM),
    "B": (AROMATIC_RIPS_TDA_CACHE, GRAPH_RIPS_DIM),
    "C": (BOND_TDA_CACHE, BOND_TDA_DIM),
    "D": (CHARGE_TDA_CACHE, CHARGE_TDA_DIM),
}


def parse_lenses(spec: str) -> list[str]:
    if not spec or spec.lower() in ("none", "vanilla", ""):
        return []
    lenses = [s.strip().upper() for s in spec.split(",") if s.strip()]
    for l in lenses:
        if l not in LENS_CACHE:
            raise ValueError(f"Unknown lens '{l}'. Choose from {list(LENS_CACHE)}.")
    return lenses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtrations", type=str, default="", help="comma list of A,B,C,D (empty = vanilla)")
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    lenses = parse_lenses(args.filtrations)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)} | lenses={lenses or ['vanilla']}")

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)
    loaders = make_loaders(
        dataset, split_idx,
        max_samples=args.max_samples,
        num_workers=4 if device.type == "cuda" else 0,
    )

    feature_bank = None
    if lenses:
        parts = []
        for l in lenses:
            cache, dim = LENS_CACHE[l]
            if not cache.exists():
                raise FileNotFoundError(f"Lens {l} cache missing: {cache}")
            parts.append(load_feature_tensor(cache, len(dataset), dim))
        feature_bank = torch.cat(parts, dim=1)
        total_dim = feature_bank.shape[1]
        model = PDGNNTDA(
            num_tasks=dataset.num_tasks,
            num_layers=NUM_BACKBONE_LAYERS,
            emb_dim=EMB_DIM,
            dropout=args.dropout,
            use_bond_tda=True,
            bond_tda_dim=total_dim,
        ).to(device)
    else:
        model = PDGNNBaseline(
            num_tasks=dataset.num_tasks,
            num_layers=NUM_BACKBONE_LAYERS,
            emb_dim=EMB_DIM,
            dropout=args.dropout,
        ).to(device)

    metrics = run_training(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        bond_tda=feature_bank,
    )

    combo = "".join(lenses) if lenses else "vanilla"
    result = {
        "model": f"PDGNN + {'+'.join(lenses)}" if lenses else "PDGNN (vanilla)",
        "filtrations": lenses,
        "lr": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        **metrics,
    }
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    drop_suffix = "" if args.dropout == DROPOUT else f"_d{args.dropout}"
    out = RESULTS_ROOT / f"multifilt_{combo}{seed_suffix}{drop_suffix}.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
