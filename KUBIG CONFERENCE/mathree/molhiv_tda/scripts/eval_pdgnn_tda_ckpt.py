#!/usr/bin/env python3
"""Evaluate a saved PDGNNTDA checkpoint on the MolHIV test set.

Reports two things so we can separate "the model actually changed" from
"the balanced-test metric is just noisy":

1. FULL (unbalanced) test ROC-AUC  -> the standard OGB metric, stable/comparable.
2. BALANCED (1:1 subsample) test ROC-AUC across several seeds -> mean/std/min/max,
   to show how much the balanced metric swings just from the random subset draw.

Example:
    python -u scripts/eval_pdgnn_tda_ckpt.py \
        --ckpt results/pdgnn_tda_3d_elec_best.pt --config pdgnn_tda_3d_elec --device cuda
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    BATCH_SIZE, BOND_TDA_DIM, BOND_TDA_CACHE, EMB_DIM, EDGE_ELECTRO_CACHE,
    MW_CACHE, NUM_BACKBONE_LAYERS, TDA_3D_CACHE, TDA_3D_DIM,
)
from data.load_molhiv import load_feature_tensor, load_molhiv, make_loaders
from models.pdgnn_tda import PDGNNTDA
from train.train_utils import evaluate_model
from utils.device import device_label, resolve_device

# Feature set of the pdgnn_tda_3d_elec config.
CFG = dict(use_bond_tda=True, use_mw=False, use_tda_3d=True, use_edge_electro=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="pdgnn_tda_3d_elec")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--balance-seeds", type=int, default=20,
                        help="How many random 1:1 subsets to evaluate for the balanced metric.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")

    dataset, split_idx, evaluator, _ = load_molhiv(str(PROJECT_ROOT / "dataset"))
    bond_tda = load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM)
    tda_3d = load_feature_tensor(TDA_3D_CACHE, len(dataset), TDA_3D_DIM)

    obj = torch.load(EDGE_ELECTRO_CACHE, weights_only=False)
    edge_phys_bank = obj["edge_phys"] if isinstance(obj, dict) else obj

    loaders = make_loaders(
        dataset, split_idx, batch_size=BATCH_SIZE, num_workers=2,
        edge_phys_bank=edge_phys_bank,
    )

    model = PDGNNTDA(
        num_tasks=dataset.num_tasks, num_layers=NUM_BACKBONE_LAYERS, emb_dim=EMB_DIM,
        dropout=args.dropout, use_bond_tda=True, bond_tda_dim=BOND_TDA_DIM,
        use_mw=False, use_tda_3d=True, tda_3d_dim=TDA_3D_DIM,
        use_edge_electro=True, edge_phys_dim=2,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    common = dict(bond_tda=bond_tda, mw=None, tda_3d=tda_3d)

    valid_full = evaluate_model(model, loaders["valid"], evaluator, device, **common)["rocauc"]
    test_full = evaluate_model(model, loaders["test"], evaluator, device, **common)["rocauc"]

    bal = []
    for s in range(args.balance_seeds):
        r = evaluate_model(
            model, loaders["test"], evaluator, device,
            balance_binary=True, balance_seed=s, **common,
        )["rocauc"]
        bal.append(r)

    print("\n===== Checkpoint evaluation =====")
    print(f"ckpt: {args.ckpt}")
    print(f"VALID full ROC-AUC            : {valid_full:.4f}")
    print(f"TEST  full (unbalanced) ROC-AUC: {test_full:.4f}   <- standard OGB, stable")
    print(f"TEST  balanced 1:1 over {len(bal)} seeds:")
    print(f"    mean {statistics.fmean(bal):.4f} | std {statistics.pstdev(bal):.4f} "
          f"| min {min(bal):.4f} | max {max(bal):.4f}")
    print("    (spread here = how much the balanced metric swings from the random subset alone)")


if __name__ == "__main__":
    main()
