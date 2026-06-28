"""GTN — Graph Transformer Network (Yun et al., NeurIPS 2019).

자동 메타패스 발견 모듈. 본 구현은 프로젝트 플랜(§4.1)에 맞춰 **타겟 노드 타입에
한정(target-restricted)**한 버전이다: 이종 엣지로부터 미리 구성한 "기저 관계"
(타겟->타겟 대칭 1-hop 메타관계 + 동종 엣지 + 항등행렬)들의 집합 A 위에서,
공식 GTN 의 GTConv/합성(곱) 메커니즘을 그대로 적용한다.

  - GTConv:   채널 c = softmax_r(W[c,r]) 로 가중합한 기저 관계 = Σ_r w_cr A_r
  - GTLayer:  첫 레이어 H = conv1(A) · conv2(A)  (길이-2 메타패스)
              이후 레이어 H = norm(H_prev) · conv(A)  (메타패스 길이 +1)
  - 채널별 GCN(X, H) 임베딩을 concat -> 선형 분류기 (Stage-1 학습 헤드)
  - 학습된 채널 인접행렬 H (C, N, N) 를 PDGNN 단계로 노출한다.

플랜 스케치(§4.2)는 forward 에서 `H @ H`(채널 자기제곱)로 적었으나, 이는 공식 GTN
(레이어마다 새 soft-combo 와의 곱)과 다르다. 충실도 원칙(repo/원논문 우선)에 따라
공식 GTN 의 합성 방식을 따른다.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
    """A_hat = A + I 를 행 정규화 (GCN 스타일). adj: (N, N)."""
    n = adj.size(0)
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return a / deg


class GTConv(nn.Module):
    """기저 관계 스택에 대한 채널별 softmax 가중합 (공식 GTN 의 1x1 conv)."""

    def __init__(self, num_relations: int, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_channels, num_relations))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: (R, N, N) -> (C, N, N)
        w = F.softmax(self.weight, dim=-1)  # (C, R)
        return torch.einsum("cr,rij->cij", w, A)

    def attention(self) -> torch.Tensor:
        return F.softmax(self.weight, dim=-1).detach()


class GTLayer(nn.Module):
    """한 GTN 레이어. first=True 면 두 conv 의 곱(길이-2), 아니면 H_prev 와의 곱(길이+1)."""

    def __init__(self, num_relations: int, num_channels: int, first: bool):
        super().__init__()
        self.first = first
        self.conv1 = GTConv(num_relations, num_channels)
        self.conv2 = GTConv(num_relations, num_channels) if first else None

    def forward(self, A: torch.Tensor, H_prev: torch.Tensor = None) -> torch.Tensor:
        if self.first:
            a = self.conv1(A)  # (C, N, N)
            b = self.conv2(A)
            return torch.bmm(a, b)
        a = self.conv1(A)
        Hn = torch.stack([_row_normalize(H_prev[c]) for c in range(H_prev.size(0))], dim=0)
        return torch.bmm(Hn, a)

    def attentions(self) -> List[torch.Tensor]:
        ws = [self.conv1.attention()]
        if self.conv2 is not None:
            ws.append(self.conv2.attention())
        return ws


class GTN(nn.Module):
    """전체 GTN: 메타패스 채널 발견 + 채널별 GCN 분류 헤드."""

    def __init__(self, num_relations: int, num_channels: int, in_dim: int,
                 hidden_dim: int, num_classes: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [GTLayer(num_relations, num_channels, first=(i == 0)) for i in range(num_layers)]
        )
        # 채널별 공유 GCN 가중치 (입력 -> hidden)
        self.gcn_weight = nn.Parameter(torch.empty(in_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gcn_weight)
        self.linear1 = nn.Linear(hidden_dim * num_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def discover(self, A: torch.Tensor) -> torch.Tensor:
        """기저 관계 스택 A: (R,N,N) -> 학습된 채널 인접행렬 H: (C,N,N)."""
        H = self.layers[0](A)
        for layer in self.layers[1:]:
            H = layer(A, H)
        return H

    def gcn_conv(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return _row_normalize(adj) @ (X @ self.gcn_weight)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """반환: (logits (N, num_classes), 학습된 채널 인접행렬 H (C, N, N))."""
        H = self.discover(A)
        embs = []
        for c in range(self.num_channels):
            embs.append(F.relu(self.gcn_conv(X, H[c])))
        Z = torch.cat(embs, dim=-1)  # (N, hidden*C)
        Z = F.dropout(Z, p=self.dropout, training=self.training)
        Z = F.relu(self.linear1(Z))
        Z = F.dropout(Z, p=self.dropout, training=self.training)
        logits = self.linear2(Z)
        return logits, H

    def channel_attentions(self) -> List[List[torch.Tensor]]:
        """레이어별 GTConv 의 (관계에 대한) softmax 가중치 — 발견된 메타패스 해석용."""
        return [layer.attentions() for layer in self.layers]
