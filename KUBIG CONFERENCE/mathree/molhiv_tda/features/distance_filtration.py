"""3D distance-based filtration utilities for PDGNN."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter


def compute_filtration_node_features_from_distances(
    edge_index: Tensor,
    dist_scores: Tensor,
    num_nodes: int,
    taus: List[float],
) -> Tensor:
    """Per-node statistics at distance thresholds (same layout as bond filtration)."""
    src, dst = edge_index[0], edge_index[1]
    src_u = torch.cat([src, dst])
    dst_u = torch.cat([dst, src])
    scores_u = torch.cat([dist_scores, dist_scores])

    feature_parts = []
    for tau in taus:
        mask = scores_u <= tau
        nbr = dst_u[mask]
        sc = scores_u[mask]

        degree = scatter(torch.ones_like(sc), nbr, dim=0, dim_size=num_nodes, reduce="sum")
        sum_score = scatter(sc, nbr, dim=0, dim_size=num_nodes, reduce="sum")
        mean_score = sum_score / degree.clamp_min(1.0)
        # placeholders for shape parity with bond stats (distance-based analogues)
        num_short = scatter((sc <= tau * 0.5).float(), nbr, dim=0, dim_size=num_nodes, reduce="sum")
        num_long = scatter((sc > tau * 0.75).float(), nbr, dim=0, dim_size=num_nodes, reduce="sum")
        feature_parts.extend([degree, sum_score, mean_score, num_short, num_long])

    return torch.stack(feature_parts, dim=1).contiguous()


class PDGNNMessageLayer(nn.Module):
    """Edge-aware message passing with SUM and MIN aggregation."""

    def __init__(self, hidden_dim: int, bond_emb_dim: int = 16, dropout: float = 0.5):
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


def compute_bond_fractions(batch, num_bond_types: int = 5) -> Tensor:
    """Graph-level bond-type fractions [num_graphs, num_bond_types]."""
    device = batch.edge_attr.device
    bond_type = batch.edge_attr[:, 0].long().clamp(0, num_bond_types - 1)
    src_graph = batch.batch[batch.edge_index[0]]
    num_graphs = int(batch.num_graphs)

    counts = torch.zeros(num_graphs, num_bond_types, device=device)
    counts.index_add_(0, src_graph, F.one_hot(bond_type, num_bond_types).float())
    return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)


def gather_edge_distances(batch, dist_bank: list[Tensor]) -> Tensor:
    """Concatenate per-graph edge distance tensors for a batched graph."""
    device = batch.x.device
    parts = []
    for g in range(batch.num_graphs):
        graph_idx = int(batch.graph_idx[g].item())
        edge_mask = batch.batch[batch.edge_index[0]] == g
        num_edges = int(edge_mask.sum().item())
        dist = dist_bank[graph_idx].to(device)
        if dist.numel() != num_edges:
            dist = torch.zeros(num_edges, device=device, dtype=torch.float32)
        parts.append(dist)
    return torch.cat(parts, dim=0)
