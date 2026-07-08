"""
PDGNN convolution layer adapted from TLC-GNN (Knowledge_Distillation/PD_conv.py).

Key PDGNN design:
- Pairwise message: lin_ij([x_i, x_j]) with LeakyReLU
- Union-Find style aggregation: concat(sum(neighbors), min(neighbors))
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def _gcn_norm(edge_index, edge_weight, num_nodes, improved=False, dtype=None):
    fill_value = 2.0 if improved else 1.0
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)

    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm


class PDConv(MessagePassing):
    """Persistence Diagram GNN convolution with sum+min aggregation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        double_input: bool = False,
        new_node_feat: bool = True,
        bond_dim: int = 0,
        normalize: bool = True,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.new_node_feat = new_node_feat
        self.bond_dim = bond_dim
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        if new_node_feat:
            in_pair = 2 * out_channels + bond_dim
            self.lin_ij = Linear(in_pair, out_channels, bias=False)

        weight_in = 2 * in_channels if double_input else in_channels
        self.weight = Parameter(torch.empty(weight_in, out_channels))

        if bias:
            self.bias = Parameter(torch.empty(2 * out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.new_node_feat:
            glorot(self.lin_ij)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_emb: Optional[Tensor] = None,
    ) -> Tensor:
        edge_weight = None
        if self.normalize:
            num_nodes = maybe_num_nodes(edge_index, x.size(0))
            edge_index, edge_weight = _gcn_norm(
                edge_index, edge_weight, num_nodes, dtype=x.dtype
            )

        x = x @ self.weight
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_emb=edge_emb,
            size=None,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_emb: Optional[Tensor],
    ) -> Tensor:
        if self.new_node_feat:
            if edge_emb is not None:
                x_j = self.lin_ij(torch.cat([x_i, x_j, edge_emb], dim=-1))
            else:
                x_j = self.lin_ij(torch.cat([x_i, x_j], dim=-1))
            x_j = F.leaky_relu(x_j, 0.2)
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr=None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        summed = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
        minimum = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="min")
        return torch.cat([summed, minimum], dim=-1)
