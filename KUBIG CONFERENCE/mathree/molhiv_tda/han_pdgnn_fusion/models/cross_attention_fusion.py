"""Node fusion and graph-wise cross-attention over fused latent vectors."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


def build_bond_pair_scale(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
    max_nodes: int,
    alpha: torch.Tensor,
    ptr: torch.Tensor | None = None,
    num_bond_types: int = 5,
) -> torch.Tensor:
    """
    Bond-type scale for each node pair within a graph.

    Returns:
        scale [batch_size, max_nodes, max_nodes] where
        scale[g, i, j] = alpha[bond_type] if nodes i,j share a bond in graph g, else 0.
    """
    device = edge_attr.device
    num_nodes = batch.size(0)
    bond_type = edge_attr[:, 0].long().clamp(0, num_bond_types - 1)
    src, dst = edge_index

    if ptr is None:
        counts = torch.bincount(batch, minlength=batch_size)
        ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        ptr[1:] = counts.cumsum(0)

    local = torch.arange(num_nodes, device=device) - ptr[batch]
    scale = torch.zeros(batch_size, max_nodes, max_nodes, device=device)

    edge_mask = batch[src] == batch[dst]
    if not edge_mask.any():
        return scale

    es, ed = src[edge_mask], dst[edge_mask]
    g = batch[es]
    ls, ld = local[es], local[ed]
    w = alpha[bond_type[edge_mask]]
    scale[g, ls, ld] = w
    scale[g, ld, ls] = w
    return scale


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, model_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


class GraphwiseCrossAttention(nn.Module):
    """
    Cross-attention where fused L attends to HAN (Z) and PDGNN (H) streams.

    Bond-modulated pairwise logits (edge-level):
      logit_ij = (Q_i · K_j / sqrt(d)) * alpha_k   when bond type k exists between i,j
      logit_ij = -inf                               when no bond (attention weight 0)
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        mode: str = "cross",
        num_bond_types: int = 5,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mode = mode
        self.num_bond_types = num_bond_types
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim

        self.q_proj = nn.Linear(model_dim, model_dim)
        if mode == "cross":
            self.k_proj = nn.Linear(2 * model_dim, model_dim)
            self.v_proj = nn.Linear(2 * model_dim, model_dim)
            self.alpha = nn.Parameter(torch.ones(num_bond_types))
        else:
            self.k_proj = nn.Linear(model_dim, model_dim)
            self.v_proj = nn.Linear(model_dim, model_dim)
            self.alpha = None

        self.z_proj = nn.Linear(model_dim, model_dim)
        self.h_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size()
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, hd = x.size()
        return x.transpose(1, 2).contiguous().view(b, n, h * hd)

    def forward(
        self,
        L: torch.Tensor,
        Z: torch.Tensor,
        H: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch.numel() == 0:
            return L

        L_pad, mask = to_dense_batch(L, batch)
        Z_pad, _ = to_dense_batch(Z, batch)
        H_pad, _ = to_dense_batch(H, batch)
        batch_size, max_nodes, _ = L_pad.shape
        if max_nodes == 0:
            return L

        device = L.device
        Q = self._split_heads(self.q_proj(L_pad))
        if self.mode == "cross":
            kv_src = torch.cat([self.z_proj(Z_pad), self.h_proj(H_pad)], dim=-1)
            K = self._split_heads(self.k_proj(kv_src))
            V = self._split_heads(self.v_proj(kv_src))
        else:
            K = self._split_heads(self.k_proj(L_pad))
            V = self._split_heads(self.v_proj(L_pad))

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        if edge_index is not None and edge_attr is not None and self.alpha is not None:
            bond_scale = build_bond_pair_scale(
                edge_index,
                edge_attr,
                batch,
                batch_size,
                max_nodes,
                self.alpha,
                ptr=ptr,
                num_bond_types=self.num_bond_types,
            )
            attn_logits = attn_logits * bond_scale.unsqueeze(1)
            bonded = pair_mask & (bond_scale > 0)
            attn_logits = attn_logits.masked_fill(~bonded.unsqueeze(1), -1e9)
        else:
            attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(1), -1e9)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        context = self._merge_heads(torch.matmul(attn, V))
        out_pad = self.out_proj(context)

        if ptr is None:
            counts = torch.bincount(batch, minlength=batch_size)
            ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
            ptr[1:] = counts.cumsum(0)
        local = torch.arange(batch.size(0), device=device) - ptr[batch]
        return out_pad[batch, local] + L


def graphwise_cross_attention(
    L: torch.Tensor,
    Z: torch.Tensor,
    H: torch.Tensor,
    batch: torch.Tensor,
    module: GraphwiseCrossAttention,
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    ptr: torch.Tensor | None = None,
) -> torch.Tensor:
    return module(L, Z, H, batch, edge_index, edge_attr, ptr=ptr)
