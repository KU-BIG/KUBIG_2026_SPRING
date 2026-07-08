"""Node fusion and graph-wise cross-attention over fused latent vectors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Nodes from different graphs cannot attend to each other (padding mask).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        mode: str = "cross",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mode = mode
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim

        self.q_proj = nn.Linear(model_dim, model_dim)
        if mode == "cross":
            self.k_proj = nn.Linear(2 * model_dim, model_dim)
            self.v_proj = nn.Linear(2 * model_dim, model_dim)
        else:
            self.k_proj = nn.Linear(model_dim, model_dim)
            self.v_proj = nn.Linear(model_dim, model_dim)

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
    ) -> torch.Tensor:
        device = L.device
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        max_nodes = 0
        for g in range(batch_size):
            max_nodes = max(max_nodes, int((batch == g).sum().item()))
        if max_nodes == 0:
            return L

        L_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        Z_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        H_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for g in range(batch_size):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            if n == 0:
                continue
            L_pad[g, :n] = L[idx]
            Z_pad[g, :n] = Z[idx]
            H_pad[g, :n] = H[idx]
            mask[g, :n] = True

        Q = self._split_heads(self.q_proj(L_pad))
        if self.mode == "cross":
            kv_src = torch.cat([self.z_proj(Z_pad), self.h_proj(H_pad)], dim=-1)
            K = self._split_heads(self.k_proj(kv_src))
            V = self._split_heads(self.v_proj(kv_src))
        else:
            K = self._split_heads(self.k_proj(L_pad))
            V = self._split_heads(self.v_proj(L_pad))

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        query_mask = mask.unsqueeze(1).unsqueeze(3)  # [B,1,N,1]
        attn_logits = attn_logits.masked_fill(~key_mask, -1e9)
        attn_logits = attn_logits.masked_fill(~query_mask, 0.0)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        context = self._merge_heads(torch.matmul(attn, V))
        out_pad = self.out_proj(context)

        out = torch.zeros_like(L)
        for g in range(batch_size):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            if n == 0:
                continue
            out[idx] = out_pad[g, :n] + L[idx]
        return out


def graphwise_cross_attention(
    L: torch.Tensor,
    Z: torch.Tensor,
    H: torch.Tensor,
    batch: torch.Tensor,
    module: GraphwiseCrossAttention,
) -> torch.Tensor:
    return module(L, Z, H, batch)
