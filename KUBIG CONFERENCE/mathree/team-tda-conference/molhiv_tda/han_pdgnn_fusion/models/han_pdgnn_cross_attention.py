"""Full HAN + PDGNN + cross-attention model."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from hetero_transform import homo_to_hetero
from models.cross_attention_fusion import FusionMLP, GraphwiseCrossAttention
from models.han_encoder import HANEncoder
from models.pdgnn_filtration_encoder import PDGNNFiltrationEncoder


def _pool(x: torch.Tensor, batch: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "add":
        return global_add_pool(x, batch)
    if mode == "max":
        return global_max_pool(x, batch)
    return global_mean_pool(x, batch)


def _extract_molecular_subgraph(data: Batch, graph_id: int) -> Data:
    node_mask = data.batch == graph_id
    node_idx = node_mask.nonzero(as_tuple=True)[0]
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    edge_index = data.edge_index[:, edge_mask]
    edge_attr = data.edge_attr[edge_mask]
    remap = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
    remap[node_idx] = torch.arange(node_idx.size(0), device=data.x.device)
    return Data(
        x=data.x[node_mask],
        edge_index=remap[edge_index],
        edge_attr=edge_attr,
    )


class HANPDGNNCrossAttentionModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        model_dim: int = 128,
        num_han_layers: int = 2,
        num_pdg_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        taus: Optional[List[float]] = None,
        bond_score_map: Optional[Dict[str, float]] = None,
        node_type_mode: str = "atomic_number",
        attention_mode: str = "cross",
        pooling: str = "mean",
        use_cross_attention: bool = True,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.node_type_mode = node_type_mode
        self.pooling = pooling
        self.use_cross_attention = use_cross_attention

        self.han_encoder = HANEncoder(hidden_dim, num_han_layers, dropout)
        self.pdgnn_encoder = PDGNNFiltrationEncoder(
            hidden_dim, num_pdg_layers, taus, bond_score_map, dropout
        )
        self.fusion = FusionMLP(2 * hidden_dim, model_dim, dropout)
        self.cross_attn = GraphwiseCrossAttention(
            model_dim, num_attention_heads, dropout, mode=attention_mode
        )
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_tasks),
        )

    def encode_han(self, data: Batch) -> torch.Tensor:
        hetero_list = getattr(data, "hetero_data", None)
        if hetero_list is None:
            z_parts = []
            for i in range(data.num_graphs):
                sub = _extract_molecular_subgraph(data, i)
                hetero, _ = homo_to_hetero(sub, node_type_mode=self.node_type_mode)
                hetero = hetero.to(data.x.device)
                z_parts.append(self.han_encoder(hetero, hetero.num_nodes))
            return torch.cat(z_parts, dim=0)

        # Batch all per-graph hetero graphs into one and run HAN once instead of
        # looping per graph (32x fewer, much larger GPU kernel launches per step).
        hetero_batch = Batch.from_data_list(hetero_list).to(data.x.device)
        typed = self.han_encoder.encode_hetero(hetero_batch)
        out = torch.zeros(data.num_nodes, self.han_encoder.hidden_dim, device=data.x.device)
        for ntype, emb in typed.items():
            store = hetero_batch[ntype]
            if not hasattr(store, "global_index"):
                continue
            abs_index = data.ptr[store.batch] + store.global_index
            out[abs_index] = emb
        return out

    def forward(self, data: Batch) -> torch.Tensor:
        Z = self.encode_han(data)
        H = self.pdgnn_encoder(data.x, data.edge_index, data.edge_attr)
        L = self.fusion(Z, H)

        if self.use_cross_attention:
            R = self.cross_attn(L, Z, H, data.batch)
        else:
            R = L

        g = _pool(R, data.batch, self.pooling)
        return self.classifier(g).view(data.num_graphs, -1)
