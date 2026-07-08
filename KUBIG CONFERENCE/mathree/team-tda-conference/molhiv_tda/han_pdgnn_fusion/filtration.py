"""Bond-structure filtration features and PDGNN-style message passing helpers."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter

from hetero_transform import bond_to_edge_type

DEFAULT_FILTRATION_SCORES: Dict[str, float] = {
    "bond_single": 1.0,
    "bond_aromatic": 1.5,
    "bond_double": 2.0,
    "bond_triple": 3.0,
    "bond_unknown": 2.0,
}


def normalize_score_map(score_map: Optional[Dict[str, float]]) -> Dict[str, float]:
    if score_map is None:
        return DEFAULT_FILTRATION_SCORES
    out = dict(DEFAULT_FILTRATION_SCORES)
    for key, val in score_map.items():
        norm_key = key if key.startswith("bond_") else f"bond_{key}"
        out[norm_key] = float(val)
    return out


def bond_filtration_score(
    edge_attr_row: torch.Tensor,
    score_map: Optional[Dict[str, float]] = None,
) -> float:
    score_map = score_map or DEFAULT_FILTRATION_SCORES
    bond_name = bond_to_edge_type(edge_attr_row)
    return float(score_map.get(bond_name, score_map["bond_unknown"]))


def compute_edge_filtration_scores(
    edge_attr: Tensor,
    score_map: Optional[Dict[str, float]] = None,
) -> Tensor:
    scores = [bond_filtration_score(edge_attr[i], score_map) for i in range(edge_attr.size(0))]
    return torch.tensor(scores, dtype=torch.float32, device=edge_attr.device)


def compute_filtration_node_features(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    taus: List[float],
    score_map: Optional[Dict[str, float]] = None,
) -> Tensor:
    """
    For each node i and threshold tau, compute local bond-filtration statistics.

    Returns tensor of shape [num_nodes, num_features_per_node].
    Features per tau (5): degree, sum_score, mean_score, num_aromatic, num_double_or_triple.
    """
    device = edge_attr.device
    filt_scores = compute_edge_filtration_scores(edge_attr, score_map)
    src, dst = edge_index[0], edge_index[1]

    bond_names = [bond_to_edge_type(edge_attr[i]) for i in range(edge_attr.size(0))]
    is_aromatic = torch.tensor(
        [1.0 if b == "bond_aromatic" else 0.0 for b in bond_names],
        device=device,
    )
    is_double_triple = torch.tensor(
        [1.0 if b in ("bond_double", "bond_triple") else 0.0 for b in bond_names],
        device=device,
    )

    # Treat edges as undirected for neighborhood stats.
    src_u = torch.cat([src, dst])
    dst_u = torch.cat([dst, src])
    scores_u = torch.cat([filt_scores, filt_scores])
    arom_u = torch.cat([is_aromatic, is_aromatic])
    dt_u = torch.cat([is_double_triple, is_double_triple])

    feature_parts = []
    for tau in taus:
        mask = scores_u <= tau
        nbr = dst_u[mask]
        sc = scores_u[mask]
        arom = arom_u[mask]
        dt = dt_u[mask]

        degree = scatter(torch.ones_like(sc), nbr, dim=0, dim_size=num_nodes, reduce="sum")
        sum_score = scatter(sc, nbr, dim=0, dim_size=num_nodes, reduce="sum")
        mean_score = sum_score / degree.clamp_min(1.0)
        num_arom = scatter(arom, nbr, dim=0, dim_size=num_nodes, reduce="sum")
        num_dt = scatter(dt, nbr, dim=0, dim_size=num_nodes, reduce="sum")
        feature_parts.extend([degree, sum_score, mean_score, num_arom, num_dt])

    return torch.stack(feature_parts, dim=1).contiguous()


class PDGNNMessageLayer(nn.Module):
    """Edge-aware message passing with SUM and MIN aggregation."""

    def __init__(self, hidden_dim: int, bond_emb_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + bond_emb_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bond_emb = nn.Embedding(5, bond_emb_dim)
        self.dropout = dropout

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        bond_type_idx: Tensor,
        filt_scores: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        bond_e = self.bond_emb(bond_type_idx.clamp(0, 4))
        msg_in = torch.cat([h[src], h[dst], bond_e, filt_scores.unsqueeze(-1)], dim=-1)
        msg = self.msg_mlp(msg_in)

        sum_msg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")
        min_msg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="min")
        degree = scatter(torch.ones_like(dst, dtype=torch.float), dst, dim=0, dim_size=h.size(0))
        has_in = degree.unsqueeze(-1) > 0
        min_msg = torch.where(has_in, min_msg, torch.zeros_like(min_msg))

        out = self.update_mlp(torch.cat([h, sum_msg, min_msg], dim=-1))
        return F.dropout(out, p=self.dropout, training=self.training)
