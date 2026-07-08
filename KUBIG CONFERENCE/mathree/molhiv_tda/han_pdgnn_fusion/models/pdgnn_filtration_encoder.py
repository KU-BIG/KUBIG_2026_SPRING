"""PDGNN-style bond filtration encoder with node-level output."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder

from filtration import PDGNNMessageLayer, compute_filtration_node_features, normalize_score_map


class PDGNNFiltrationEncoder(nn.Module):
    """
    Bond-structure filtration encoder.

    Returns h_pdg of shape [num_nodes_in_batch, hidden_dim] where h_pdg[i] = h_i^K.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        taus: Optional[List[float]] = None,
        bond_score_map: Optional[Dict[str, float]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.taus = taus or [1.0, 1.5, 2.0, 3.0]
        self.bond_score_map = normalize_score_map(bond_score_map)

        stats_per_tau = 5
        in_filtration_dim = len(self.taus) * stats_per_tau
        self.filtration_mlp = nn.Sequential(
            nn.Linear(in_filtration_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            [PDGNNMessageLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        filt_feats = compute_filtration_node_features(
            edge_index,
            edge_attr,
            num_nodes,
            self.taus,
            self.bond_score_map,
        ).to(x.device)
        h = self.filtration_mlp(filt_feats) + self.atom_encoder(x)

        bond_type_idx = edge_attr[:, 0].long()
        from filtration import compute_edge_filtration_scores

        filt_scores = compute_edge_filtration_scores(edge_attr, self.bond_score_map).to(x.device)

        for layer in self.layers:
            h = layer(h, edge_index, bond_type_idx, filt_scores)
        return h
