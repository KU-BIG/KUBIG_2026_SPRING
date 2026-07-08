"""PDGNN (3D distance filtration) + bond-type weighted MW fusion."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from features.distance_filtration import compute_bond_fractions, gather_edge_distances
from models.bond_weighted_fusion import BondTypeWeightedFusion
from models.pdgnn_3d_dist_encoder import PDGNN3DDistEncoder


class PDGNN3DDistMW(nn.Module):
    def __init__(
        self,
        num_tasks: int = 1,
        emb_dim: int = 300,
        num_layers: int = 4,
        taus: Optional[List[float]] = None,
        dropout: float = 0.5,
        num_bond_types: int = 5,
    ):
        super().__init__()
        self.encoder = PDGNN3DDistEncoder(emb_dim, num_layers, taus, dropout)
        self.fusion = BondTypeWeightedFusion(emb_dim, num_bond_types)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_tasks),
        )

    def forward(
        self,
        batch,
        edge_dist_bank: list[torch.Tensor],
        mw: torch.Tensor,
    ) -> torch.Tensor:
        edge_dist = gather_edge_distances(batch, edge_dist_bank)
        g_pdg = self.encoder(batch.x, batch.edge_index, batch.edge_attr, edge_dist, batch.batch)
        bond_frac = compute_bond_fractions(batch)
        fused, _ = self.fusion(g_pdg, mw, bond_frac)
        return self.head(fused)
