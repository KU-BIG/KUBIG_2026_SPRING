"""PDGNN baseline model for OGBG-MolHIV graph classification."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.inits import reset

from models.pdgnn_conv import PDConv


class PDGNNBaseline(torch.nn.Module):
    """
    PDGNN backbone for molecular graphs.

    Architecture follows TLC-GNN Teacher_model.Base_Model (type='PDGNN'):
    5 PDConv layers with sum+min aggregation, adapted for OGB atom/bond encoders.
    """

    def __init__(
        self,
        num_tasks: int = 1,
        num_layers: int = 5,
        emb_dim: int = 300,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.1)

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        # PDGNN stack: each layer output is 2 * emb_dim due to sum+min aggregation
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            PDConv(emb_dim, emb_dim, double_input=False, bond_dim=emb_dim, normalize=False)
        )
        for _ in range(1, num_layers):
            self.convs.append(
                PDConv(emb_dim, emb_dim, double_input=True, bond_dim=emb_dim, normalize=False)
            )

        self.graph_pred = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * emb_dim, num_tasks),
        )

    def encode(self, batch) -> torch.Tensor:
        x = self.atom_encoder(batch.x)
        edge_emb = self.bond_encoder(batch.edge_attr)

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, batch.edge_index, edge_emb=edge_emb)
            x = self.prelu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return global_add_pool(x, batch.batch)

    def forward(self, batch):
        return self.graph_pred(self.encode(batch))

    def reset_parameters(self):
        reset(self)
