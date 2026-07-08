"""Semantic Attention Fusion — 여러 채널의 위상 특징을 의미 어텐션으로 융합.

HAN(Wang et al., WWW 2019) 의 semantic-level attention 을 그대로 따른다:
  - 채널 c 의 특징을 비선형 변환 후, 학습 가능한 쿼리 q 와 내적 -> 채널 중요도 w_c
  - beta = softmax_c(w_c) -> fused = Σ_c beta_c * topo_c

이로써 PDGNN 이 채널(=발견된 메타패스)별로 만든 위상 특징을 하나로 합친다.
플랜에서 concat 대신 attention fusion 을 채택하기로 한 부분을 구현한다.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SemanticAttentionFusion(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, channel_feats: List[torch.Tensor]) -> torch.Tensor:
        """channel_feats: 길이 C 리스트, 각 (N, in_dim). 반환: (N, in_dim) 융합 특징."""
        if len(channel_feats) == 1:
            return channel_feats[0]
        stacked = torch.stack(channel_feats, dim=1)  # (N, C, in_dim)
        w = self.query(self.project(stacked)).squeeze(-1)  # (N, C)
        # 채널 중요도는 모든 노드에 공유되는 의미수준 어텐션 (HAN): 노드 평균 후 softmax
        beta = torch.softmax(w.mean(dim=0), dim=0)  # (C,)
        fused = torch.einsum("c,ncd->nd", beta, stacked)
        self.last_beta = beta.detach()
        return fused
