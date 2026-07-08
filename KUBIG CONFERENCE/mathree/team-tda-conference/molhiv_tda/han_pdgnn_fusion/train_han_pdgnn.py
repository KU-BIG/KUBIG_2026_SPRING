#!/usr/bin/env python3
"""Train HAN+PDGNN fusion models on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import get_dataset, make_dataloaders  # noqa: E402
from models.baselines import build_model  # noqa: E402
from utils import load_config, resolve_device, save_checkpoint, set_seed  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.view(batch.num_graphs, -1).cpu())
        y_pred.append(pred.view(batch.num_graphs, -1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def _sync_optimizer_params(optimizer, model):
    """Add params for lazily-created submodules (e.g. per-relation HAN convs)
    discovered during forward(), which are missing from the optimizer because
    it was built from model.parameters() before any forward pass ran."""
    tracked = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in tracked]
    if new_params:
        optimizer.add_param_group({"params": new_params})


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        is_labeled = batch.y == batch.y
        loss = F.binary_cross_entropy_with_logits(
            pred.to(torch.float32)[is_labeled],
            batch.y.to(torch.float32)[is_labeled],
        )
        loss.backward()
        _sync_optimizer_params(optimizer, model)
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


def run_training(cfg: dict, model_name: str, args):
    set_seed(args.seed)
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    dataset_root = (PROJECT_ROOT / cfg["dataset_root"]).resolve()
    dataset, split_idx, evaluator, _ = get_dataset(dataset_root, cfg["node_type_mode"])
    loaders = make_dataloaders(
        dataset,
        split_idx,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        node_type_mode=cfg["node_type_mode"],
        max_samples=args.max_samples,
        cache_root=PROJECT_ROOT / "cache",
    )

    model = build_model(model_name, cfg, num_tasks=dataset.num_tasks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_valid = -1.0
    best_test = -1.0
    best_state = None
    stale = 0
    patience = cfg.get("patience", 10)

    results_dir = PROJECT_ROOT / cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / f"han_pdgnn_{model_name}_best.pt"

    epoch_bar = tqdm(range(1, cfg["epochs"] + 1), desc="epoch")
    for epoch in epoch_bar:
        loss = train_one_epoch(model, loaders["train"], optimizer, device)
        valid_auc = evaluate(model, loaders["valid"], evaluator, device)
        test_auc = evaluate(model, loaders["test"], evaluator, device)

        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = test_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(ckpt_path, model, optimizer, epoch, {"valid_rocauc": valid_auc})
            stale = 0
        else:
            stale += 1

        epoch_bar.set_postfix(
            loss=f"{loss:.4f}",
            valid=f"{valid_auc:.4f}",
            test=f"{test_auc:.4f}",
            best_valid=f"{best_valid:.4f}",
            stale=stale,
        )

        if stale >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "model": f"han_pdgnn_{model_name}",
        "valid_rocauc": best_valid,
        "test_rocauc": best_test,
        "config": cfg,
    }
    out_json = results_dir / f"han_pdgnn_{model_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(result)
    print(f"Saved results to {out_json}")
    return result


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Apply CLI overrides on top of config.yaml."""
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train HAN+PDGNN fusion models on OGBG-MolHIV")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="main",
        choices=["gcn", "han_only", "pdgnn_only", "concat", "main"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--device", type=str, default=None, help="Override config device (auto/cpu/cuda)")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr")
    parser.add_argument("--dropout", type=float, default=None, help="Override config dropout")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    cfg = apply_cli_overrides(cfg, args)
    run_training(cfg, args.model, args)


if __name__ == "__main__":
    main()
