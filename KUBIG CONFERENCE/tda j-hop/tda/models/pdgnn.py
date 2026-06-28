"""PDGNN — 그래프의 1차원 extended persistence diagram(EPD)을 근사하는 신경망.

원본: TLC-GNN `Knowledge_Distillation/pdgnn_modern.py` (Yan et al., NeurIPS 2022 의
PyG 2.x 충실 재구현). 본 파일은 공유/Colab 호환을 위해 `torch_scatter` 의존성을
torch 기본 scatter 연산으로 대체했을 뿐 집계(SUM ⊕ MIN)는 수치적으로 동일하다.

구조 (논문 §4.2):
  - 입력: 노드별 필터값 f(v) ∈ R.
  - L 개 PDGNN 레이어, 각 레이어:
      h_u = combine( SUM_v MSG(h_v), MIN_v MSG(h_v), h_u W_self )
      MSG(h_v) = alpha_uv * PReLU( (h_u || h_v) W )
  - 각 엣지 (u,v): edge_mlp(h_u || h_v) -> (birth, death).
EPD 의 정답은 gudhi 로 계산한 exact diagram 이며, **학습 라벨로만** 쓰인다(추론 시 정확
계산 없음). 이는 PDGNN 이 neural EPD 근사기라는 정의에 따른 것이다.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import MessagePassing


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.size(-1)))
    out.scatter_add_(0, index.view(-1, 1).expand_as(src), src)
    return out


def _scatter_min(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_full((dim_size, src.size(-1)), float("inf"))
    out.scatter_reduce_(0, index.view(-1, 1).expand_as(src), src, reduce="amin", include_self=True)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


class PDGNNLayer(MessagePassing):
    """PDGNN 메시지 패싱 레이어. agg='sum_min'(기본, 원본) 또는 'sum'(ablation D2)."""

    def __init__(self, in_dim: int, out_dim: int, agg: str = "sum_min"):
        super().__init__(aggr=None, flow="source_to_target")
        self.agg = agg
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_msg = nn.Linear(2 * in_dim, out_dim, bias=False)
        n_agg = 3 if agg == "sum_min" else 2  # [sum(,min) || self]
        self.lin_combine = nn.Linear(n_agg * out_dim, out_dim, bias=True)
        self.act_msg = nn.PReLU(out_dim)
        self.act_out = nn.PReLU(out_dim)

    def forward(self, x, edge_index, edge_weight: Optional[torch.Tensor] = None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, dim_size=x.size(0))
        combined = torch.cat([out, self.lin_self(x)], dim=-1)
        return self.act_out(self.lin_combine(combined))

    def message(self, x_i, x_j, edge_weight):
        msg = self.act_msg(self.lin_msg(torch.cat([x_i, x_j], dim=-1)))
        return edge_weight.view(-1, 1) * msg

    def aggregate(self, inputs, index, dim_size: Optional[int] = None):
        sum_agg = _scatter_sum(inputs, index, dim_size)
        if self.agg == "sum_min":
            min_agg = _scatter_min(inputs, index, dim_size)
            return torch.cat([sum_agg, min_agg], dim=-1)
        return sum_agg


class PDGNN(nn.Module):
    """모든 엣지에 대해 1차원 extended persistence (birth, death) 쌍을 예측."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.0,
                 agg: str = "sum_min"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(PDGNNLayer(1, hidden_dim, agg=agg))
        for _ in range(num_layers - 1):
            self.layers.append(PDGNNLayer(hidden_dim, hidden_dim, agg=agg))
        self.dropout = dropout
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, filt_value, edge_index, edge_weight=None, pred_edges=None):
        x = filt_value
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if pred_edges is None:
            pred_edges = edge_index
        h_u = x[pred_edges[0]]
        h_v = x[pred_edges[1]]
        return self.edge_mlp(torch.cat([h_u, h_v], dim=-1))


def bipartite_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """예측 EPD(pred, (E,2)) 와 정답(gt, (K,2)) 간 Hungarian 매칭 거리.

    원본 `train_pdgnn_lp._bipartite_loss` 와 동일.
    """
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    cost = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(dim=-1)  # (E, K)
    cost_np = cost.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    matched = cost[row_idx, col_idx]
    return matched.mean()
