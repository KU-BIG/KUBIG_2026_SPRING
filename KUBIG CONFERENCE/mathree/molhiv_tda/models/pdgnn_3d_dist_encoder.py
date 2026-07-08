"""PDGNN encoder with 3D conformer distance filtration."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool

from features.distance_filtration import (
    PDGNNMessageLayer,
    compute_filtration_node_features_from_distances,
)


class PDGNN3DDistEncoder(nn.Module):
    """
    PDGNN-style encoder using 3D inter-atomic distances as edge filtration scores.

    Returns graph-level embedding g_pdg.
    """

    def __init__(
        self,
        emb_dim: int = 300,
        num_layers: int = 4,
        taus: Optional[List[float]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.taus = taus or [1.5, 2.0, 2.5, 3.0, 4.0]
        self.dropout = dropout

        in_filtration_dim = len(self.taus) * 5
        self.filtration_mlp = nn.Sequential(
            nn.Linear(in_filtration_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.atom_encoder = AtomEncoder(emb_dim)
        self.layers = nn.ModuleList(
            [PDGNNMessageLayer(emb_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_dist: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        filt_feats = compute_filtration_node_features_from_distances(
            edge_index, edge_dist, num_nodes, self.taus
        )
        h = self.filtration_mlp(filt_feats) + self.atom_encoder(x)

        bond_type_idx = edge_attr[:, 0].long()
        for layer in self.layers:
            h = layer(h, edge_index, bond_type_idx, edge_dist)

        return global_add_pool(h, batch)
