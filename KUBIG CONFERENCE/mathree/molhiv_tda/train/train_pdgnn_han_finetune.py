#!/usr/bin/env python3
"""Fine-tune a frozen PDGNN(+TDA) backbone with a bond-relation HAN node branch.

The PDGNN backbone is loaded from a checkpoint produced by train_pdgnn_tda.py
(--save-ckpt) and fully frozen. Only the HAN branch, the fusion gate, and the new
head are trained. See models/pdgnn_han_finetune.py for the architecture.

Example (after creating results/pdgnn_tda_3d_elec_best.pt):
    python -u train/train_pdgnn_han_finetune.py \
        --config pdgnn_han_finetune_3d_elec \
        --backbone-ckpt results/pdgnn_tda_3d_elec_best.pt \
        --lr 1e-4 --dropout 0.3 --weight-decay 1e-5 \
        --device cuda --out results/pdgnn_han_finetune_3d_elec.json
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
    BATCH_SIZE,
    BOND_TDA_DIM,
    BOND_TDA_CACHE,
    DEFAULT_DEVICE,
    DROPOUT,
    EMB_DIM,
    EDGE_ELECTRO_CACHE,
    EPOCHS,
    GRAPH_RIPS_DIM,
    GRAPH_RIPS_TDA_CACHE,
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
from models.pdgnn_han_finetune import PDGNNHANFinetune
from train.train_utils import run_training, save_result
from utils.device import device_label, resolve_device

# Fine-tuning configs mirror the pretrained backbone's feature set so the frozen
# backbone state_dict loads cleanly and receives the same batch inputs.
CONFIGS = {
    "pdgnn_han_finetune_3d_elec": dict(
        use_bond_tda=True,
        use_mw=False,
        use_tda_3d=True,
        use_edge_electro=True,
        use_graph_tda=False,
        balance_test=True,
        label="PDGNN(frozen) + HAN node fusion + BondTDA + 3DTDA + ElectroEdge",
    ),
    # Same as above + multifiltration lenses A (full-graph hop-Rips) and B
    # (aromatic-subgraph hop-Rips) concatenated into the head (graph_tda channel).
    "pdgnn_han_finetune_3d_elec_multifilt": dict(
        use_bond_tda=True,
        use_mw=False,
        use_tda_3d=True,
        use_edge_electro=True,
        use_graph_tda=True,
        balance_test=True,
        label="PDGNN(frozen) + HAN node fusion + BondTDA + 3DTDA + ElectroEdge + Multifiltration(A+B)",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pdgnn_han_finetune_3d_elec",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--backbone-ckpt", type=str, required=True,
                        help="Path to the pretrained PDGNN(+TDA) state_dict (.pt).")
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--han-hidden", type=int, default=128)
    parser.add_argument("--han-layers", type=int, default=2)
    parser.add_argument("--han-heads", type=int, default=4)
    parser.add_argument("--han-dropout", type=float, default=0.2)
    parser.add_argument("--balanced-train", action="store_true",
                        help="Train with ~1:1 pos/neg per epoch (minority oversampling).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: results/{config}.json).")
    parser.add_argument("--save-ckpt", type=str, default=None,
                        help="If set, save the best fine-tuned model state_dict here.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")
    cfg = CONFIGS[args.config]

    if not Path(args.backbone_ckpt).exists():
        raise FileNotFoundError(
            f"Missing backbone checkpoint {args.backbone_ckpt}. "
            "Create it first via train/train_pdgnn_tda.py --save-ckpt."
        )

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)

    bond_tda = load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM) if cfg["use_bond_tda"] else None
    mw = load_feature_tensor(MW_CACHE, len(dataset), 1) if cfg["use_mw"] else None
    tda_3d = load_feature_tensor(TDA_3D_CACHE, len(dataset), TDA_3D_DIM) if cfg["use_tda_3d"] else None

    if cfg["use_bond_tda"] and not BOND_TDA_CACHE.exists():
        raise FileNotFoundError(f"Missing {BOND_TDA_CACHE}. Run scripts/preprocess_bond_tda.py first.")
    if cfg["use_tda_3d"] and not TDA_3D_CACHE.exists():
        raise FileNotFoundError(f"Missing {TDA_3D_CACHE}. Run scripts/preprocess_3d_tda.py first.")

    # Multifiltration lenses A (full-graph hop-Rips) + B (aromatic hop-Rips).
    graph_tda = None
    graph_tda_dim = 0
    if cfg.get("use_graph_tda", False):
        for cache in (GRAPH_RIPS_TDA_CACHE, AROMATIC_RIPS_TDA_CACHE):
            if not cache.exists():
                raise FileNotFoundError(
                    f"Missing {cache}. Run scripts/preprocess_graph_rips_tda.py first."
                )
        lens_a = load_feature_tensor(GRAPH_RIPS_TDA_CACHE, len(dataset), GRAPH_RIPS_DIM)
        lens_b = load_feature_tensor(AROMATIC_RIPS_TDA_CACHE, len(dataset), GRAPH_RIPS_DIM)
        graph_tda = torch.cat([lens_a, lens_b], dim=1)
        graph_tda_dim = graph_tda.shape[1]

    edge_phys_bank = None
    if cfg["use_edge_electro"]:
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
        balanced_train=args.balanced_train,
    )

    model = PDGNNHANFinetune(
        num_tasks=dataset.num_tasks,
        num_layers=NUM_BACKBONE_LAYERS,
        emb_dim=EMB_DIM,
        dropout=args.dropout,
        use_bond_tda=cfg["use_bond_tda"],
        bond_tda_dim=BOND_TDA_DIM,
        use_mw=cfg["use_mw"],
        use_tda_3d=cfg["use_tda_3d"],
        tda_3d_dim=TDA_3D_DIM,
        use_edge_electro=cfg["use_edge_electro"],
        edge_phys_dim=2,
        use_graph_tda=cfg.get("use_graph_tda", False),
        graph_tda_dim=graph_tda_dim,
        han_hidden=args.han_hidden,
        han_layers=args.han_layers,
        han_heads=args.han_heads,
        han_dropout=args.han_dropout,
        backbone_ckpt=args.backbone_ckpt,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable:,} | Frozen (backbone) params: {n_frozen:,}")

    metrics = run_training(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bond_tda=bond_tda,
        mw=mw,
        tda_3d=tda_3d,
        graph_tda=graph_tda,
        balance_test=cfg["balance_test"],
        test_balance_seed=args.seed,
        save_ckpt=args.save_ckpt,
    )

    result = {
        "model": cfg["label"],
        "backbone": "PDGNN (frozen)",
        "backbone_ckpt": args.backbone_ckpt,
        "bond_type_tda": cfg["use_bond_tda"],
        "molecular_weight": cfg["use_mw"],
        "tda_3d": cfg["use_tda_3d"],
        "electro_edge": cfg["use_edge_electro"],
        "multifiltration": cfg.get("use_graph_tda", False),
        "graph_tda_dim": graph_tda_dim,
        "balanced_test_eval": cfg["balance_test"],
        "balanced_train": args.balanced_train,
        "config": args.config,
        "lr": args.lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "han_hidden": args.han_hidden,
        "han_layers": args.han_layers,
        "han_heads": args.han_heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "trainable_params": n_trainable,
        "frozen_params": n_frozen,
        **metrics,
    }
    out = Path(args.out) if args.out else RESULTS_ROOT / f"{args.config}.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
