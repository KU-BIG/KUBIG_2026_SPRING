"""RGCN — Relational GCN (Schlichtkrull et al., ESWC 2018,
"Modeling Relational Data with Graph Convolutional Networks").

관계(edge type)마다 별도 weight matrix W_r 로 메시지를 변환하고 합산한다. self-connection
은 PyG `RGCNConv` 의 root_weight(W_0)로 처리(원논문의 self-loop 관계와 동일 역할).
basis decomposition(num_bases)으로 관계별 파라미터를 공유할 수 있다(원논문 §2.2).

본 파이프라인에서는 HAN 과 *비교 가능*하도록 **타겟-타겟 기저 관계**(bundle.base_relations)를
관계 타입으로 사용한다(target-restricted). 입력 특징 x 는 HAN 과 동일(원본 또는 [원본⊕위상]).
참고 구현: PyTorch Geometric `RGCNConv` (원논문 충실 구현)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_relations: int,
                 num_bases=None, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(RGCNConv(in_dim, num_classes, num_relations, num_bases=num_bases))
        else:
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            self.convs.append(RGCNConv(hidden_dim, num_classes, num_relations, num_bases=num_bases))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
