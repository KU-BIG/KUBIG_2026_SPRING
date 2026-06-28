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


def _gtn_norm(H: torch.Tensor, add: bool) -> torch.Tensor:
    """공식 GTN 정규화(Yun et al. 2019, model.py `norm`): H^T 위에서 대각을 0으로 비우고
    (add 면 1로 채움) 행 정규화한 뒤 다시 전치한다. 학습된 메타패스 인접 H 는 비대칭이므로
    이 전치가 전파 방향과 어느 차수(in/out)로 정규화할지를 결정한다. H: (N, N)."""
    Ht = H.t()
    n = Ht.size(0)
    eye = torch.eye(n, device=H.device, dtype=H.dtype)
    Ht = Ht * (eye == 0)        # 대각 제거
    if add:
        Ht = Ht + eye          # 자기 루프(최종 GCN conv 에만)
    deg = Ht.sum(dim=1, keepdim=True)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    Ht = deg_inv * Ht
    return Ht.t()


class GTConv(nn.Module):
    """기저 관계 스택에 대한 채널별 softmax 가중합 (공식 GTN 의 1x1 conv)."""

    def __init__(self, num_relations: int, num_channels: int):
        super().__init__()
        # 공식 GTN: 상수 초기화로 관계축 softmax 가 균등에서 출발(편향 없는 평균에서 학습 시작).
        self.weight = nn.Parameter(torch.empty(num_channels, num_relations))
        nn.init.constant_(self.weight, 0.1)

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
        # 레이어 사이 정규화: 공식과 동일하게 self-loop 없이(add=False)
        Hn = torch.stack([_gtn_norm(H_prev[c], add=False) for c in range(H_prev.size(0))], dim=0)
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
        # 공식 GTN gcn_conv: norm(H, add=True) 후 H^T 로 전파 -> rownorm(H^T, diag=1) @ (X W)
        return _gtn_norm(adj, add=True).t() @ (X @ self.gcn_weight)

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
