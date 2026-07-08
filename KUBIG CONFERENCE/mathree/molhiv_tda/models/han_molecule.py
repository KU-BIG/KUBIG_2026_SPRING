"""HAN-style molecular model using bond types as heterogeneous relations."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, global_mean_pool

BOND_NAMES = {
    0: "single",
    1: "double",
    2: "triple",
    3: "aromatic",
    4: "misc",
}


def pyg_data_to_bond_heterodata(data) -> HeteroData:
    """Convert a molecular PyG graph into bond-type HeteroData."""
    hetero = HeteroData()
    hetero["atom"].x = data.x
    bond_type = data.edge_attr[:, 0]

    for idx, name in BOND_NAMES.items():
        mask = bond_type == idx
        if mask.sum() == 0:
            continue
        hetero["atom", name, "atom"].edge_index = data.edge_index[:, mask]
        hetero["atom", name, "atom"].edge_attr = data.edge_attr[mask]

    hetero.y = data.y
    if hasattr(data, "graph_idx"):
        hetero.graph_idx = data.graph_idx
    return hetero


def hetero_batch_to_dicts(batch_hetero):
    """Extract x_dict and edge_index_dict from a batched HeteroData."""
    x_dict = {node_type: batch_hetero[node_type].x for node_type in batch_hetero.node_types}
    edge_index_dict = {
        edge_type: batch_hetero[edge_type].edge_index
        for edge_type in batch_hetero.edge_types
    }
    return x_dict, edge_index_dict


class HANMolecule(torch.nn.Module):
    """Relation-aware HAN over bond-type edge relations."""

    def __init__(
        self,
        metadata,
        in_channels: int = 9,
        hidden_channels: int = 128,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tasks: int = 1,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HANConv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                )
            )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + extra_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_tasks),
        )

    def encode(self, batch_hetero):
        x_dict = {node_type: self.atom_encoder(batch_hetero[node_type].x) for node_type in batch_hetero.node_types}
        edge_index_dict = {
            edge_type: batch_hetero[edge_type].edge_index
            for edge_type in batch_hetero.edge_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(F.dropout(v, p=self.dropout, training=self.training)) for k, v in x_dict.items()}
        atom_batch = batch_hetero["atom"].batch
        return global_mean_pool(x_dict["atom"], atom_batch)

    def forward(self, batch_hetero, extra_features: torch.Tensor | None = None):
        graph_emb = self.encode(batch_hetero)
        if extra_features is not None:
            graph_emb = torch.cat([graph_emb, extra_features], dim=1)
        return self.head(graph_emb)
