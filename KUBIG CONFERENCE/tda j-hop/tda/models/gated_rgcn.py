"""Gated-RGCN — RGCN 메시지에 PEGN 식 위상 게이트를 곱하는 변형 (실험 ①, docs 참조).

PEGN(Zhao et al., AISTATS 2020)의 "persistence 로 메시지 재가중" 아이디어를 RGCN 에 이식:
  m_{u←v, r} = (W_r h_v) ⊙ gate(g_u, g_v),  gate = σ(MLP([g_u, g_v]))
게이트 입력 g 는 노드별 위상 특징(persistence image 융합 벡터). gate_feat=None 이면 게이트≡1
(= plain RGCN 과 동일 구조) — base/concat 조건도 이 클래스로 돌려 구현 차이를 제거한다.

주의: PEGN 원논문은 GCN(동종)+edge-vicinity PI 이며, 관계별 W_r·노드 PI 게이트는 우리 어댑테이션.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_agg(msg: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, msg.size(1), device=msg.device, dtype=msg.dtype)
    out.index_add_(0, dst, msg)
    cnt = torch.zeros(n, device=msg.device, dtype=msg.dtype)
    cnt.index_add_(0, dst, torch.ones(dst.size(0), device=msg.device, dtype=msg.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


class GatedRelConv(nn.Module):
    """관계별 W_r + (옵션) 위상 게이트. RGCNConv(aggr=mean, root_weight)와 동일 골격."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 gate_dim: Optional[int] = None, gate_hidden: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.root = nn.Linear(in_dim, out_dim)
        self.gate_mlp = None
        if gate_dim is not None:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * gate_dim, gate_hidden), nn.ELU(),
                nn.Linear(gate_hidden, out_dim))

    def forward(self, x, edge_index, edge_type, gate_feat=None):
        n = x.size(0)
        out = self.root(x)
        for r in range(self.weight.size(0)):
            m = edge_type == r
            if not bool(m.any()):
                continue
            src, dst = edge_index[0, m], edge_index[1, m]
            msg = x[src] @ self.weight[r]                      # (E_r, out)
            if self.gate_mlp is not None and gate_feat is not None:
                g = torch.sigmoid(self.gate_mlp(
                    torch.cat([gate_feat[dst], gate_feat[src]], dim=-1)))
                msg = msg * g
            out = out + _mean_agg(msg, dst, n)
        return out


class GatedRGCN(nn.Module):
    """2층 기본. gate_feat 미제공 시 plain RGCN 과 동일 계산(게이트 경로 미사용)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_relations: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 gate_dim: Optional[int] = None, gate_hidden: int = 64):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList([
            GatedRelConv(dims[i], dims[i + 1], num_relations,
                         gate_dim=gate_dim, gate_hidden=gate_hidden)
            for i in range(num_layers)])

    def forward(self, x, edge_index, edge_type, gate_feat=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, gate_feat=gate_feat)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
