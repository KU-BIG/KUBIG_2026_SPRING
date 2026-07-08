"""Baseline models for ablation comparison."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from models.han_pdgnn_cross_attention import HANPDGNNCrossAttentionModel, _pool
from models.pdgnn_filtration_encoder import PDGNNFiltrationEncoder


class GCNBaseline(nn.Module):
    """Simple GCN baseline (not the main model)."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2, num_tasks: int = 1):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, data.batch)
        return self.head(g).view(data.num_graphs, -1)


class HANOnlyModel(nn.Module):
    """Baseline 2: HAN encoder only."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_han_layers: int = 2,
        dropout: float = 0.2,
        node_type_mode: str = "atomic_number",
        pooling: str = "mean",
        num_tasks: int = 1,
    ):
        super().__init__()
        self.core = HANPDGNNCrossAttentionModel(
            hidden_dim=hidden_dim,
            model_dim=hidden_dim,
            num_han_layers=num_han_layers,
            num_pdg_layers=1,
            node_type_mode=node_type_mode,
            pooling=pooling,
            use_cross_attention=False,
            num_tasks=num_tasks,
        )
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        Z = self.core.encode_han(data)
        g = _pool(Z, data.batch, self.pooling)
        return self.head(g).view(data.num_graphs, -1)


class PDGNNOnlyModel(nn.Module):
    """Baseline 3: PDGNN filtration encoder only."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_pdg_layers: int = 3,
        taus: Optional[List[float]] = None,
        bond_score_map: Optional[Dict[str, float]] = None,
        dropout: float = 0.2,
        pooling: str = "mean",
        num_tasks: int = 1,
    ):
        super().__init__()
        self.encoder = PDGNNFiltrationEncoder(hidden_dim, num_pdg_layers, taus, bond_score_map, dropout)
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        H = self.encoder(data.x, data.edge_index, data.edge_attr)
        g = _pool(H, data.batch, self.pooling)
        return self.head(g).view(data.num_graphs, -1)


class HANPDGNNConcatModel(HANPDGNNCrossAttentionModel):
    """Baseline 4: concat fusion, no cross-attention."""

    def __init__(self, **kwargs):
        kwargs["use_cross_attention"] = False
        super().__init__(**kwargs)


def build_model(name: str, cfg: dict, num_tasks: int = 1) -> nn.Module:
    common = dict(
        hidden_dim=cfg["hidden_dim"],
        model_dim=cfg["model_dim"],
        num_han_layers=cfg["num_han_layers"],
        num_pdg_layers=cfg["num_pdg_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        dropout=cfg["dropout"],
        taus=cfg["filtration_taus"],
        bond_score_map=cfg.get("bond_filtration_scores"),
        node_type_mode=cfg["node_type_mode"],
        attention_mode=cfg["attention_mode"],
        pooling=cfg.get("pooling", "mean"),
        num_tasks=num_tasks,
    )
    name = name.lower()
    if name in ("gcn", "gcn_baseline"):
        return GCNBaseline(cfg["hidden_dim"], cfg["num_pdg_layers"], cfg["dropout"], num_tasks)
    if name in ("han", "han_only"):
        return HANOnlyModel(
            hidden_dim=cfg["hidden_dim"],
            num_han_layers=cfg["num_han_layers"],
            dropout=cfg["dropout"],
            node_type_mode=cfg["node_type_mode"],
            pooling=cfg.get("pooling", "mean"),
            num_tasks=num_tasks,
        )
    if name in ("pdgnn", "pdgnn_only"):
        return PDGNNOnlyModel(
            hidden_dim=cfg["hidden_dim"],
            num_pdg_layers=cfg["num_pdg_layers"],
            taus=cfg["filtration_taus"],
            bond_score_map=cfg.get("bond_filtration_scores"),
            dropout=cfg["dropout"],
            pooling=cfg.get("pooling", "mean"),
            num_tasks=num_tasks,
        )
    if name in ("concat", "han_pdgnn_concat"):
        return HANPDGNNConcatModel(**common)
    if name in ("main", "han_pdgnn_cross", "cross_attention"):
        return HANPDGNNCrossAttentionModel(**common, use_cross_attention=True)
    raise ValueError(f"Unknown model name: {name}")
