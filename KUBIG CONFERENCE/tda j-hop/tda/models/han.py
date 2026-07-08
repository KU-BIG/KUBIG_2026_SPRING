"""HAN — Heterogeneous Graph Attention Network (Wang et al., WWW 2019), downstream 분류기.

타겟 노드 타입 한 종류 위에서 메타패스 그래프들(PAP, PSP, ...)을 엣지 타입으로 갖는
이종 그래프에 PyG `HANConv` 를 적용한다. HANConv 가 노드수준 어텐션 + 메타패스
의미수준 어텐션을 모두 수행한다(HAN 의 핵심).

입력 노드 특징은 [원본 특징 ⊕ (옵션) 융합된 위상 특징] 으로 강화될 수 있다.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 target: str, metapath_edge_types: List[Tuple[str, str, str]],
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.target = target
        self.metapath_edge_types = metapath_edge_types
        metadata = ([target], list(metapath_edge_types))
        self.han = HANConv(in_dim, hidden_dim, metadata, heads=heads, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        x_dict = {self.target: x}
        out = self.han(x_dict, edge_index_dict)[self.target]
        out = F.elu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.classifier(out)
