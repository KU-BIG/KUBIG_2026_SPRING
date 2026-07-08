"""PDGNN model augmented with TDA and molecular weight features."""
from __future__ import annotations

import torch

from models.pdgnn_baseline import PDGNNBaseline


class PDGNNTDA(torch.nn.Module):
    """
    PDGNN backbone + optional concatenation of:
    - bond-type TDA vector
    - molecular weight scalar
    - 3D TDA vector
    """

    def __init__(
        self,
        num_tasks: int = 1,
        num_layers: int = 5,
        emb_dim: int = 300,
        dropout: float = 0.5,
        use_bond_tda: bool = False,
        bond_tda_dim: int = 50,
        use_mw: bool = False,
        use_tda_3d: bool = False,
        tda_3d_dim: int = 75,
        use_edge_electro: bool = False,
        edge_phys_dim: int = 2,
    ):
        super().__init__()
        self.use_bond_tda = use_bond_tda
        self.use_mw = use_mw
        self.use_tda_3d = use_tda_3d
        self.use_edge_electro = use_edge_electro

        self.backbone = PDGNNBaseline(
            num_tasks=0,
            num_layers=num_layers,
            emb_dim=emb_dim,
            dropout=dropout,
            edge_phys_dim=edge_phys_dim if use_edge_electro else 0,
        )

        graph_dim = 2 * emb_dim
        extra_dim = 0
        if use_bond_tda:
            extra_dim += bond_tda_dim
        if use_mw:
            extra_dim += 1
        if use_tda_3d:
            extra_dim += tda_3d_dim

        self.head = torch.nn.Sequential(
            torch.nn.Linear(graph_dim + extra_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * emb_dim, num_tasks),
        )

    def forward(
        self,
        batch,
        bond_tda: torch.Tensor | None = None,
        mw: torch.Tensor | None = None,
        tda_3d: torch.Tensor | None = None,
    ):
        graph_emb = self.backbone.encode(batch)
        extras = [graph_emb]

        if self.use_bond_tda:
            if bond_tda is None:
                raise ValueError("bond_tda features required but not provided")
            extras.append(bond_tda)

        if self.use_mw:
            if mw is None:
                raise ValueError("molecular weight features required but not provided")
            extras.append(mw)

        if self.use_tda_3d:
            if tda_3d is None:
                raise ValueError("3D TDA features required but not provided")
            extras.append(tda_3d)

        return self.head(torch.cat(extras, dim=1))

    def reset_parameters(self):
        self.backbone.reset_parameters()
        for layer in self.head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
