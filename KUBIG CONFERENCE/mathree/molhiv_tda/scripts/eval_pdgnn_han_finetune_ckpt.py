#!/usr/bin/env python3
"""Evaluate a saved PDGNN(frozen)+HAN fine-tuning checkpoint on MolHIV.

Reports:
1. VALID full ROC-AUC (real distribution).
2. TEST full (unbalanced) ROC-AUC  -> standard OGB metric, stable.
3. TEST balanced 1:1 ROC-AUC over many seeds -> mean/std/min/max (the noisy metric).

The checkpoint must be a full PDGNNHANFinetune state_dict (backbone + HAN + gate +
head), e.g. produced by train_pdgnn_han_finetune.py --save-ckpt.

Example:
    python -u scripts/eval_pdgnn_han_finetune_ckpt.py \
        --ckpt results/pdgnn_han_finetune_best.pt \
        --han-hidden 128 --dropout 0.3 --device cuda --balance-seeds 30
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
    AROMATIC_RIPS_TDA_CACHE, BATCH_SIZE, BOND_TDA_DIM, BOND_TDA_CACHE, EMB_DIM,
    EDGE_ELECTRO_CACHE, GRAPH_RIPS_DIM, GRAPH_RIPS_TDA_CACHE,
    NUM_BACKBONE_LAYERS, TDA_3D_CACHE, TDA_3D_DIM,
)
from data.load_molhiv import load_feature_tensor, load_molhiv, make_loaders
from models.pdgnn_han_finetune import PDGNNHANFinetune
from train.train_utils import evaluate_model
from utils.device import device_label, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--han-hidden", type=int, default=128)
    parser.add_argument("--han-layers", type=int, default=2)
    parser.add_argument("--han-heads", type=int, default=4)
    parser.add_argument("--han-dropout", type=float, default=0.2)
    parser.add_argument("--balance-seeds", type=int, default=30)
    parser.add_argument("--multifilt", action="store_true",
                        help="Checkpoint includes multifiltration lenses A+B (graph_tda).")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")

    dataset, split_idx, evaluator, _ = load_molhiv(str(PROJECT_ROOT / "dataset"))
    bond_tda = load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM)
    tda_3d = load_feature_tensor(TDA_3D_CACHE, len(dataset), TDA_3D_DIM)
    obj = torch.load(EDGE_ELECTRO_CACHE, weights_only=False)
    edge_phys_bank = obj["edge_phys"] if isinstance(obj, dict) else obj

    graph_tda = None
    graph_tda_dim = 0
    if args.multifilt:
        lens_a = load_feature_tensor(GRAPH_RIPS_TDA_CACHE, len(dataset), GRAPH_RIPS_DIM)
        lens_b = load_feature_tensor(AROMATIC_RIPS_TDA_CACHE, len(dataset), GRAPH_RIPS_DIM)
        graph_tda = torch.cat([lens_a, lens_b], dim=1)
        graph_tda_dim = graph_tda.shape[1]

    loaders = make_loaders(
        dataset, split_idx, batch_size=BATCH_SIZE, num_workers=2,
        edge_phys_bank=edge_phys_bank,
    )

    # backbone_ckpt=None: skip loading backbone from disk; the full fine-tune
    # state_dict (which includes the backbone) is loaded right after.
    model = PDGNNHANFinetune(
        num_tasks=dataset.num_tasks, num_layers=NUM_BACKBONE_LAYERS, emb_dim=EMB_DIM,
        dropout=args.dropout, use_bond_tda=True, bond_tda_dim=BOND_TDA_DIM,
        use_mw=False, use_tda_3d=True, tda_3d_dim=TDA_3D_DIM,
        use_edge_electro=True, edge_phys_dim=2,
        use_graph_tda=args.multifilt, graph_tda_dim=graph_tda_dim,
        han_hidden=args.han_hidden, han_layers=args.han_layers,
        han_heads=args.han_heads, han_dropout=args.han_dropout,
        backbone_ckpt=None,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    common = dict(bond_tda=bond_tda, mw=None, tda_3d=tda_3d, graph_tda=graph_tda)
    valid_full = evaluate_model(model, loaders["valid"], evaluator, device, **common)["rocauc"]
    test_full = evaluate_model(model, loaders["test"], evaluator, device, **common)["rocauc"]

    bal = []
    for s in range(args.balance_seeds):
        bal.append(evaluate_model(
            model, loaders["test"], evaluator, device,
            balance_binary=True, balance_seed=s, **common,
        )["rocauc"])

    print("\n===== Fine-tune checkpoint evaluation =====")
    print(f"ckpt: {args.ckpt}")
    print(f"VALID full ROC-AUC             : {valid_full:.4f}")
    print(f"TEST  full (unbalanced) ROC-AUC : {test_full:.4f}   <- standard OGB, stable")
    print(f"TEST  balanced 1:1 over {len(bal)} seeds:")
    print(f"    mean {statistics.fmean(bal):.4f} | std {statistics.pstdev(bal):.4f} "
          f"| min {min(bal):.4f} | max {max(bal):.4f}")


if __name__ == "__main__":
    main()
