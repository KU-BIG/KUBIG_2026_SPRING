"""HAN-style heterogeneous graph encoder with node-level output."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import softmax, scatter

from hetero_transform import hetero_node_embeddings_to_homo


class RelationAttentionConv(nn.Module):
    """Node-level attention aggregation for one heterogeneous relation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.att_src = nn.Linear(hidden_dim, 1, bias=False)
        self.att_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(x_dst.size(0), x_dst.size(1), device=x_dst.device)

        src, dst = edge_index
        msg = self.msg_lin(x_src[src])
        att = self.att_src(x_src[src]) + self.att_dst(x_dst[dst])
        att = softmax(att.squeeze(-1), dst, num_nodes=x_dst.size(0))
        out = scatter(msg * att.unsqueeze(-1), dst, dim=0, dim_size=x_dst.size(0), reduce="sum")
        return F.dropout(out, p=self.dropout, training=self.training)


class HANLayer(nn.Module):
    """One HAN layer: relation-specific attention + semantic-level attention."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_convs = nn.ModuleDict()
        self.semantic_query = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def _get_relation_conv(self, rel_name: str, device: torch.device) -> RelationAttentionConv:
        key = rel_name.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "").replace("'", "")
        if key not in self.relation_convs:
            self.relation_convs[key] = RelationAttentionConv(self.hidden_dim, self.dropout).to(device)
        return self.relation_convs[key]

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict) -> Dict[str, torch.Tensor]:
        device = next(iter(x_dict.values())).device
        # Collect per-node-type relation outputs before semantic fusion.
        node_rel_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in x_dict}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_name, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            conv = self._get_relation_conv(str(edge_type), device)
            out_dst = conv(x_dict[src_type], x_dict[dst_type], edge_index.to(device))
            node_rel_out[dst_type].append(out_dst)

        out_dict: Dict[str, torch.Tensor] = {}
        for ntype, x in x_dict.items():
            rel_outs = node_rel_out.get(ntype, [])
            if not rel_outs:
                out_dict[ntype] = x
                continue
            # Semantic-level attention over relation outputs at each node.
            stacked = torch.stack(rel_outs, dim=0)  # [R, N_t, D]
            query = self.semantic_query(x).unsqueeze(0)  # [1, N_t, D]
            scores = (stacked * query).sum(dim=-1) / (self.hidden_dim ** 0.5)  # [R, N_t]
            beta = torch.softmax(scores, dim=0)
            fused = (beta.unsqueeze(-1) * stacked).sum(dim=0)
            out_dict[ntype] = F.relu(x + fused)
        return out_dict


class HANEncoder(nn.Module):
    """
    HAN-style encoder returning node embeddings z_i aligned with homogeneous node order.

    Output shape: [num_nodes_in_batch, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.layers = nn.ModuleList([HANLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = dropout

    def encode_hetero(self, hetero: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {nt: self.atom_encoder(hetero[nt].x) for nt in hetero.node_types}
        device = next(iter(x_dict.values())).device
        edge_index_dict = {et: hetero[et].edge_index.to(device) for et in hetero.edge_types}
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict

    def forward(self, hetero: HeteroData, num_nodes: int) -> torch.Tensor:
        typed = self.encode_hetero(hetero)
        device = next(iter(typed.values())).device
        return hetero_node_embeddings_to_homo(typed, hetero, num_nodes, device)


def batch_hetero_from_data_list(data_list: List[HeteroData]) -> HeteroData:
    return Batch.from_data_list(data_list)
