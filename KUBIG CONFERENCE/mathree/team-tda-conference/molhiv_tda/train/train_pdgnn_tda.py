#!/usr/bin/env python3
"""Train PDGNN with TDA / molecular weight ablations on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    BATCH_SIZE,
    BOND_TDA_DIM,
    BOND_TDA_CACHE,
    DEFAULT_DEVICE,
    DIST_FILTRATION_TAUS,
    DROPOUT,
    EMB_DIM,
    EDGE_DIST_CACHE,
    EPOCHS,
    MW_CACHE,
    NUM_BACKBONE_LAYERS,
    NUM_WORKERS,
    PATIENCE,
    RESULTS_ROOT,
    TDA_3D_CACHE,
    TDA_3D_DIM,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import load_feature_tensor, load_molhiv, make_loaders
from models.pdgnn_3d_dist_mw import PDGNN3DDistMW
from models.pdgnn_tda import PDGNNTDA
from train.train_utils import load_edge_dist_bank, run_training, save_result
from utils.device import device_label, resolve_device

CONFIGS = {
    "pdgnn_mw": dict(model="pdgnn_tda", use_bond_tda=False, use_mw=True, use_tda_3d=False, label="PDGNN + MW"),
    "pdgnn_bond_tda": dict(model="pdgnn_tda", use_bond_tda=True, use_mw=False, use_tda_3d=False, label="PDGNN + BondTDA"),
    "pdgnn_bond_tda_mw": dict(model="pdgnn_tda", use_bond_tda=True, use_mw=True, use_tda_3d=False, label="PDGNN + BondTDA + MW"),
    "pdgnn_3d_tda": dict(model="pdgnn_tda", use_bond_tda=False, use_mw=False, use_tda_3d=True, label="PDGNN + 3DTDA"),
    "pdgnn_bond_tda_3d_mw": dict(
        model="pdgnn_tda",
        use_bond_tda=True,
        use_mw=True,
        use_tda_3d=True,
        label="PDGNN + BondTDA + 3DTDA + MW",
    ),
    "pdgnn_3d_dist_mw": dict(
        model="pdgnn_3d_dist_mw",
        use_bond_tda=False,
        use_mw=True,
        use_tda_3d=False,
        use_edge_dist=True,
        label="PDGNN (3D dist filtration) + bond-weighted MW",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")
    cfg = CONFIGS[args.config]

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)
    loaders = make_loaders(
        dataset, split_idx,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers if device.type == "cuda" else 0,
    )

    bond_tda = load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM) if cfg.get("use_bond_tda") else None
    mw = load_feature_tensor(MW_CACHE, len(dataset), 1) if cfg.get("use_mw") else None
    tda_3d = load_feature_tensor(TDA_3D_CACHE, len(dataset), TDA_3D_DIM) if cfg.get("use_tda_3d") else None
    edge_dist_bank = None

    if cfg.get("use_bond_tda") and not BOND_TDA_CACHE.exists():
        raise FileNotFoundError(f"Missing {BOND_TDA_CACHE}. Run scripts/preprocess_bond_tda.py first.")
    if cfg.get("use_mw") and not MW_CACHE.exists():
        raise FileNotFoundError(f"Missing {MW_CACHE}. Run scripts/preprocess_molecular_weight.py first.")
    if cfg.get("use_tda_3d") and not TDA_3D_CACHE.exists():
        raise FileNotFoundError(f"Missing {TDA_3D_CACHE}. Run scripts/preprocess_3d_tda.py first.")
    if cfg.get("use_edge_dist"):
        if not EDGE_DIST_CACHE.exists():
            raise FileNotFoundError(
                f"Missing {EDGE_DIST_CACHE}. Run scripts/preprocess_3d_edge_dist.py first."
            )
        edge_dist_bank = load_edge_dist_bank(EDGE_DIST_CACHE)

    if cfg["model"] == "pdgnn_3d_dist_mw":
        model = PDGNN3DDistMW(
            num_tasks=dataset.num_tasks,
            emb_dim=EMB_DIM,
            num_layers=NUM_BACKBONE_LAYERS,
            taus=DIST_FILTRATION_TAUS,
            dropout=args.dropout,
        ).to(device)
    else:
        model = PDGNNTDA(
            num_tasks=dataset.num_tasks,
            num_layers=NUM_BACKBONE_LAYERS,
            emb_dim=EMB_DIM,
            dropout=args.dropout,
            use_bond_tda=cfg.get("use_bond_tda", False),
            bond_tda_dim=BOND_TDA_DIM,
            use_mw=cfg.get("use_mw", False),
            use_tda_3d=cfg.get("use_tda_3d", False),
            tda_3d_dim=TDA_3D_DIM,
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
        bond_tda=bond_tda,
        mw=mw,
        tda_3d=tda_3d,
        edge_dist_bank=edge_dist_bank,
    )

    result = {
        "model": cfg["label"],
        "backbone": "PDGNN",
        "bond_type_tda": cfg.get("use_bond_tda", False),
        "molecular_weight": cfg.get("use_mw", False),
        "tda_3d": cfg.get("use_tda_3d", False),
        "edge_dist_filtration": cfg.get("use_edge_dist", False),
        "lr": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        **metrics,
    }
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    out = RESULTS_ROOT / f"{args.config}{seed_suffix}.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
