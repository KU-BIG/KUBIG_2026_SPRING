#!/usr/bin/env python3
"""Train PDGNN baseline on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    BATCH_SIZE,
    DEFAULT_DEVICE,
    DROPOUT,
    EMB_DIM,
    EPOCHS,
    EDGE_ELECTRO_CACHE,
    NUM_BACKBONE_LAYERS,
    NUM_WORKERS,
    PATIENCE,
    RESULTS_ROOT,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import load_molhiv, make_loaders
from models.pdgnn_baseline import PDGNNBaseline
from train.train_utils import run_training, save_result
from utils.device import device_label, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="cuda, cpu, or auto (default: auto)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Use subset for quick dev runs")
    parser.add_argument(
        "--use-electro-edge",
        action="store_true",
        help="Use cached edge physics [distance, coulomb] from RDKit.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)
    edge_phys_bank = None
    if args.use_electro_edge:
        if not EDGE_ELECTRO_CACHE.exists():
            raise FileNotFoundError(
                f"Missing {EDGE_ELECTRO_CACHE}. Run scripts/preprocess_edge_electrostatic.py first."
            )
        obj = torch.load(EDGE_ELECTRO_CACHE, weights_only=False)
        edge_phys_bank = obj["edge_phys"] if isinstance(obj, dict) else obj
    loaders = make_loaders(
        dataset, split_idx,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        edge_phys_bank=edge_phys_bank,
    )

    model = PDGNNBaseline(
        num_tasks=dataset.num_tasks,
        num_layers=NUM_BACKBONE_LAYERS,
        emb_dim=EMB_DIM,
        dropout=DROPOUT,
        edge_phys_dim=2 if args.use_electro_edge else 0,
    ).to(device)

    metrics = run_training(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    result = {
        "model": "PDGNN",
        "backbone": "PDGNN",
        "bond_type_tda": False,
        "molecular_weight": False,
        "tda_3d": False,
        "electro_edge": args.use_electro_edge,
        **metrics,
    }
    out = RESULTS_ROOT / "pdgnn_baseline.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
