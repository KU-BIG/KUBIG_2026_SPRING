#!/usr/bin/env python3
"""Train PDGNN baseline on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import functools

import torch

from config import (
    BATCH_SIZE,
    DEFAULT_DEVICE,
    DROPOUT,
    EMB_DIM,
    EPOCHS,
    NUM_BACKBONE_LAYERS,
    NUM_WORKERS,
    PATIENCE,
    RESULTS_ROOT,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import load_molhiv, make_loaders
from models.pdgnn_baseline import PDGNNBaseline
from train.train_utils import focal_loss_with_logits, run_training, save_result
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
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)
    loaders = make_loaders(
        dataset, split_idx,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers if device.type == "cuda" else 0,
    )

    model = PDGNNBaseline(
        num_tasks=dataset.num_tasks,
        num_layers=NUM_BACKBONE_LAYERS,
        emb_dim=EMB_DIM,
        dropout=args.dropout,
    ).to(device)

    loss_fn = None
    if args.loss == "focal":
        loss_fn = functools.partial(
            focal_loss_with_logits, gamma=args.focal_gamma, alpha=args.focal_alpha
        )

    metrics = run_training(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        loss_fn=loss_fn,
    )

    result = {
        "model": "PDGNN",
        "backbone": "PDGNN",
        "bond_type_tda": False,
        "molecular_weight": False,
        "tda_3d": False,
        "lr": args.lr,
        "dropout": args.dropout,
        "loss": args.loss,
        "seed": args.seed,
        **metrics,
    }
    suffix = "" if args.loss == "bce" else f"_{args.loss}"
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    out = RESULTS_ROOT / f"pdgnn_lr{args.lr}_dropout{args.dropout}{suffix}{seed_suffix}.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
