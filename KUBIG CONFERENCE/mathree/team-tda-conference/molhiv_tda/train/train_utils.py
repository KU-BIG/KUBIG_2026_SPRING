"""Shared training and evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from torch_geometric.data import Batch
from tqdm.auto import tqdm


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


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017), operating on raw logits."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    evaluator: Evaluator,
    device,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        pred = _forward_model(
            model, batch, bond_tda, mw, tda_3d, edge_dist_bank, device
        )

        y_true.append(batch.y.view(batch.y.size(0), -1).detach().cpu())
        y_pred.append(pred.view(pred.size(0), -1).detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return {"rocauc": evaluator.eval(input_dict)["rocauc"]}


def _forward_model(
    model,
    batch,
    bond_tda,
    mw,
    tda_3d,
    edge_dist_bank,
    device,
):
    if edge_dist_bank is not None:
        mw_b = gather_graph_features(batch, mw.to(device))
        return model(batch, edge_dist_bank, mw_b)

    kwargs = {}
    if bond_tda is not None:
        kwargs["bond_tda"] = gather_graph_features(batch, bond_tda.to(device))
    if mw is not None:
        kwargs["mw"] = gather_graph_features(batch, mw.to(device))
    if tda_3d is not None:
        kwargs["tda_3d"] = gather_graph_features(batch, tda_3d.to(device))
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
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
    loss_fn=None,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    loss_fn = loss_fn or F.binary_cross_entropy_with_logits

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = _forward_model(
            model, batch, bond_tda, mw, tda_3d, edge_dist_bank, device
        )

        is_labeled = batch.y == batch.y
        loss = loss_fn(
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
    edge_dist_bank: Optional[list[torch.Tensor]] = None,
    loss_fn=None,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid = -1.0
    best_test = -1.0
    best_state = None
    stale = 0

    epoch_bar = tqdm(range(epochs), desc="epoch")
    for _ in epoch_bar:
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
            edge_dist_bank=edge_dist_bank,
            loss_fn=loss_fn,
        )
        valid_score = evaluate_model(
            model,
            loaders["valid"],
            evaluator,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
            edge_dist_bank=edge_dist_bank,
        )["rocauc"]
        test_score = evaluate_model(
            model,
            loaders["test"],
            evaluator,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
            edge_dist_bank=edge_dist_bank,
        )["rocauc"]

        if valid_score > best_valid:
            best_valid = valid_score
            best_test = test_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            valid=f"{valid_score:.4f}",
            best_valid=f"{best_valid:.4f}",
            stale=stale,
        )

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"valid_rocauc": best_valid, "test_rocauc": best_test}


def save_result(result: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

