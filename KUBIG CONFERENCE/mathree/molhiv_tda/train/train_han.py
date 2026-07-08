#!/usr/bin/env python3
"""Train HAN-style bond-relation model on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from torch_geometric.loader import DataLoader

from config import (
    BATCH_SIZE,
    BOND_TDA_CACHE,
    BOND_TDA_DIM,
    DROPOUT,
    EPOCHS,
    MW_CACHE,
    PATIENCE,
    RESULTS_ROOT,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import IndexedGraphDataset, load_feature_tensor, load_molhiv
from models.han_molecule import HANMolecule, pyg_data_to_bond_heterodata
from train.train_utils import gather_graph_features, save_result


def hetero_collate(items):
    from torch_geometric.data import Batch

    hetero_items = [pyg_data_to_bond_heterodata(item) for item in items]
    return Batch.from_data_list(hetero_items)


@torch.no_grad()
def evaluate_han(model, loader, evaluator, device, extra_features=None):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        extra = None
        if extra_features is not None:
            extra = gather_graph_features(batch, extra_features.to(device))
        pred = model(batch, extra_features=extra)
        y_true.append(batch.y.view(batch.num_graphs, -1).cpu())
        y_pred.append(pred.view(batch.num_graphs, -1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def train_han(
    model,
    loaders,
    evaluator,
    device,
    epochs,
    patience,
    lr,
    weight_decay,
    extra_features=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid = -1.0
    best_test = -1.0
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            extra = None
            if extra_features is not None:
                extra = gather_graph_features(batch, extra_features.to(device))
            pred = model(batch, extra_features=extra)
            is_labeled = batch.y == batch.y
            loss = F.binary_cross_entropy_with_logits(
                pred.to(torch.float32)[is_labeled],
                batch.y.to(torch.float32)[is_labeled],
            )
            loss.backward()
            optimizer.step()

        valid = evaluate_han(model, loaders["valid"], evaluator, device, extra_features)
        test = evaluate_han(model, loaders["test"], evaluator, device, extra_features)
        if valid > best_valid:
            best_valid = valid
            best_test = test
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"valid_rocauc": best_valid, "test_rocauc": best_test}


def build_extra_features(config, dataset):
    parts = []
    if "bond" in config:
        parts.append(load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM))
    if "mw" in config:
        parts.append(load_feature_tensor(MW_CACHE, len(dataset), 1))
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="han",
        choices=["han", "han_bond_tda", "han_bond_tda_mw", "han_mw"],
    )
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)

    sample = pyg_data_to_bond_heterodata(dataset[0])
    metadata = sample.metadata()

    extra = build_extra_features(args.config, dataset)
    model = HANMolecule(
        metadata=metadata,
        hidden_channels=128,
        heads=4,
        num_layers=2,
        dropout=DROPOUT,
        num_tasks=dataset.num_tasks,
        extra_dim=0 if extra is None else extra.shape[1],
    ).to(device)

    loaders = {}
    for split in ("train", "valid", "test"):
        subset = IndexedGraphDataset(dataset, split_idx[split].tolist())
        loaders[split] = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=hetero_collate,
        )

    metrics = train_han(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        extra_features=extra,
    )

    result = {
        "model": args.config,
        "bond_relation_han": True,
        "bond_type_tda": "bond" in args.config,
        "molecular_weight": "mw" in args.config,
        **metrics,
    }
    out = RESULTS_ROOT / f"{args.config}.json"
    save_result(result, out)
    print(result)


if __name__ == "__main__":
    main()
