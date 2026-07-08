"""Gated-HAN — HAN 의 메타패스 내 노드 attention 을 위상 게이트로 변조 (실험 ①-HAN).

구조 (우리 HAN(han.py: HANConv 1층+분류기)과 대응):
  메타패스 m 마다: GAT 식 attention α_uv → 메시지에 (옵션) 위상 게이트 ⊙σ(MLP([g_u,g_v])) →
  head concat → 메타패스 간 semantic attention 융합 → 분류기.
gate_feat=None 이면 게이트 미사용(=plain attention) — base/concat 조건도 동일 구현으로 돌려
구현 차이를 제거한다. PEGN(AISTATS 2020)의 "persistence 로 메시지 재가중"의 HAN 이식.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tda.models.fusion import SemanticAttentionFusion


def _segment_softmax(e: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """목적지(dst)별 softmax. e: (E, H)."""
    m = torch.full((n, e.size(1)), -1e9, device=e.device, dtype=e.dtype)
    m = m.scatter_reduce(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True)
    ex = torch.exp(e - m[dst])
    s = torch.zeros(n, e.size(1), device=e.device, dtype=e.dtype)
    s.index_add_(0, dst, ex)
    return ex / s[dst].clamp(min=1e-12)


class GatedHAN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_metapaths: int,
                 heads: int = 4, dropout: float = 0.5,
                 gate_dim: Optional[int] = None, gate_hidden: int = 64):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads, self.per = heads, hidden_dim // heads
        self.dropout = dropout
        self.W = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False)
                                for _ in range(num_metapaths)])
        self.att_src = nn.Parameter(torch.empty(num_metapaths, heads, self.per))
        self.att_dst = nn.Parameter(torch.empty(num_metapaths, heads, self.per))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.gate_mlp = None
        if gate_dim is not None:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * gate_dim, gate_hidden), nn.ELU(), nn.Linear(gate_hidden, heads))
        self.semantic = SemanticAttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mp_edge_indices: List[torch.Tensor],
                gate_feat: torch.Tensor = None) -> torch.Tensor:
        n = x.size(0)
        outs = []
        xin = F.dropout(x, p=self.dropout, training=self.training)
        for m, ei in enumerate(mp_edge_indices):
            h = self.W[m](xin).view(n, self.heads, self.per)
            src, dst = ei[0], ei[1]
            e = F.leaky_relu((h[src] * self.att_src[m]).sum(-1)
                             + (h[dst] * self.att_dst[m]).sum(-1), 0.2)     # (E, H)
            alpha = _segment_softmax(e, dst, n)
            msg = h[src] * alpha.unsqueeze(-1)
            if self.gate_mlp is not None and gate_feat is not None:
                g = torch.sigmoid(self.gate_mlp(
                    torch.cat([gate_feat[dst], gate_feat[src]], dim=-1)))    # (E, H)
                msg = msg * g.unsqueeze(-1)
            out = torch.zeros(n, self.heads, self.per, device=x.device, dtype=msg.dtype)
            out.index_add_(0, dst, msg)
            outs.append(F.elu(out.reshape(n, self.heads * self.per)))
        z = self.semantic(outs)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)
