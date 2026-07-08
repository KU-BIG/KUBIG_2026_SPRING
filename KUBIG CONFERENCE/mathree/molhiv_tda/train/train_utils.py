"""Shared training and evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from torch_geometric.data import Batch


def gather_graph_features(
    batch,
    feature_tensor: torch.Tensor,
) -> torch.Tensor:
    """Select precomputed graph-level features for a batch."""
    idx = batch.graph_idx.view(-1)
    return feature_tensor[idx]


def load_edge_dist_bank(path: Path) -> list[torch.Tensor]:
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, dict) and "distances" in obj:
        return obj["distances"]
    return obj


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    evaluator: Evaluator,
    device,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
    graph_tda: Optional[torch.Tensor] = None,
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
    balance_binary: bool = False,
    balance_seed: int = 0,
) -> Dict[str, float]:
    model.eval()
    # Move feature banks to device once (no-op if already there); avoids
    # copying the full bank to GPU on every batch.
    bond_tda = bond_tda.to(device) if bond_tda is not None else None
    mw = mw.to(device) if mw is not None else None
    tda_3d = tda_3d.to(device) if tda_3d is not None else None
    graph_tda = graph_tda.to(device) if graph_tda is not None else None
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        pred = _forward_model(
            model, batch, bond_tda, mw, tda_3d, edge_dist_bank, graph_tda
        )

        y_true.append(batch.y.view(batch.y.size(0), -1).detach().cpu())
        y_pred.append(pred.view(pred.size(0), -1).detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if balance_binary:
        y_true, y_pred = _balanced_binary_subset(y_true, y_pred, seed=balance_seed)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return {"rocauc": evaluator.eval(input_dict)["rocauc"]}


def _balanced_binary_subset(
    y_true,
    y_pred,
    seed: int = 0,
):
    # OGB MolHIV labels are [N, 1] with {0,1}
    labels = y_true.reshape(-1)
    pos = [i for i, v in enumerate(labels) if v == 1]
    neg = [i for i, v in enumerate(labels) if v == 0]
    if not pos or not neg:
        return y_true, y_pred
    k = min(len(pos), len(neg))
    g = torch.Generator()
    g.manual_seed(seed)
    pos_idx = torch.randperm(len(pos), generator=g)[:k].tolist()
    neg_idx = torch.randperm(len(neg), generator=g)[:k].tolist()
    keep = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    keep = torch.tensor(keep, dtype=torch.long)[torch.randperm(2 * k, generator=g)].tolist()
    return y_true[keep], y_pred[keep]


def _forward_model(
    model,
    batch,
    bond_tda,
    mw,
    tda_3d,
    edge_dist_bank,
    graph_tda=None,
):
    """Feature banks are expected to already live on the batch's device."""
    if edge_dist_bank is not None:
        mw_b = gather_graph_features(batch, mw)
        return model(batch, edge_dist_bank, mw_b)

    kwargs = {}
    if bond_tda is not None:
        kwargs["bond_tda"] = gather_graph_features(batch, bond_tda)
    if mw is not None:
        kwargs["mw"] = gather_graph_features(batch, mw)
    if tda_3d is not None:
        kwargs["tda_3d"] = gather_graph_features(batch, tda_3d)
    if graph_tda is not None:
        kwargs["graph_tda"] = gather_graph_features(batch, graph_tda)
    if kwargs:
        return model(batch, **kwargs)
    return model(batch)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
    graph_tda: Optional[torch.Tensor] = None,
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
) -> float:
    model.train()
    # Move feature banks to device once (no-op if already there).
    bond_tda = bond_tda.to(device) if bond_tda is not None else None
    mw = mw.to(device) if mw is not None else None
    tda_3d = tda_3d.to(device) if tda_3d is not None else None
    graph_tda = graph_tda.to(device) if graph_tda is not None else None
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = _forward_model(
            model, batch, bond_tda, mw, tda_3d, edge_dist_bank, graph_tda
        )

        is_labeled = batch.y == batch.y
        loss = F.binary_cross_entropy_with_logits(
            pred.to(torch.float32)[is_labeled],
            batch.y.to(torch.float32)[is_labeled],
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


def run_training(
    model,
    loaders,
    evaluator,
    device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
    graph_tda: Optional[torch.Tensor] = None,
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
    balance_test: bool = False,
    test_balance_seed: int = 0,
    save_ckpt: Optional[Path] = None,
) -> Dict[str, float]:
    # Only optimize params that require grad, so frozen backbones (fine-tuning)
    # are left untouched even though they are part of model.parameters().
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    # Hoist feature banks onto the device once so per-epoch calls never re-copy.
    bond_tda = bond_tda.to(device) if bond_tda is not None else None
    mw = mw.to(device) if mw is not None else None
    tda_3d = tda_3d.to(device) if tda_3d is not None else None
    graph_tda = graph_tda.to(device) if graph_tda is not None else None
    best_valid = -1.0
    best_test = -1.0
    best_state = None
    stale = 0

    for epoch in range(epochs):
        train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
            graph_tda=graph_tda,
            edge_dist_bank=edge_dist_bank,
        )
        valid_score = evaluate_model(
            model,
            loaders["valid"],
            evaluator,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
            graph_tda=graph_tda,
            edge_dist_bank=edge_dist_bank,
        )["rocauc"]

        if valid_score > best_valid:
            # Only pay for a test-set pass when validation actually improves.
            test_score = evaluate_model(
                model,
                loaders["test"],
                evaluator,
                device,
                bond_tda=bond_tda,
                mw=mw,
                tda_3d=tda_3d,
                graph_tda=graph_tda,
                edge_dist_bank=edge_dist_bank,
                balance_binary=balance_test,
                balance_seed=test_balance_seed + epoch,
            )["rocauc"]
            best_valid = valid_score
            best_test = test_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_ckpt is not None:
            save_ckpt = Path(save_ckpt)
            save_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_ckpt)
            print(f"Saved best model state_dict to {save_ckpt}")

    return {"valid_rocauc": best_valid, "test_rocauc": best_test}


def save_result(result: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

