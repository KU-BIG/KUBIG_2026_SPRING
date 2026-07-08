"""작은 합성 이종 그래프 — 스모크 테스트 및 "데이터셋 추가" 예시.

ACM 과 동일한 노드/엣지 타입 명명(paper/author/subject/term, paper-cite/ref-paper,
paper-to-author 등)을 써서 같은 config 스키마로 돌아간다. 클래스는 subject 와
약하게 연관되도록 심어(planted) 학습 가능성을 준다(스모크 용도이며 성능 주장 아님).
"""
from __future__ import annotations

from typing import Tuple

import torch


def load_toy(data_root: str = "./data", seed: int = 0) -> Tuple[object, str]:
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    n_paper, n_author, n_subject, n_term, feat, n_class = 60, 25, 3, 10, 16, 3
    d = HeteroData()
    d["paper"].x = torch.randn(n_paper, feat, generator=g)
    d["author"].x = torch.randn(n_author, feat, generator=g)
    d["subject"].x = torch.randn(n_subject, feat, generator=g)
    d["term"].num_nodes = n_term  # term 은 ACM 처럼 특징 없음 -> num_nodes 명시

    # 각 paper 를 한 subject 에 배정 -> 라벨 = subject (planted)
    paper_subject = torch.randint(0, n_subject, (n_paper,), generator=g)
    y = paper_subject.clone()
    d["paper"].y = y

    def bip(n_src, n_dst, deg):
        src = torch.arange(n_src).repeat_interleave(deg)
        dst = torch.randint(0, n_dst, (n_src * deg,), generator=g)
        return torch.stack([src, dst], dim=0)

    pa = bip(n_paper, n_author, 2)
    d["paper", "to", "author"].edge_index = pa
    d["author", "to", "paper"].edge_index = pa[[1, 0]]
    ps = torch.stack([torch.arange(n_paper), paper_subject], dim=0)
    d["paper", "to", "subject"].edge_index = ps
    d["subject", "to", "paper"].edge_index = ps[[1, 0]]
    pt = bip(n_paper, n_term, 2)
    d["paper", "to", "term"].edge_index = pt
    d["term", "to", "paper"].edge_index = pt[[1, 0]]
    # paper-paper cite/ref: 같은 subject 끼리 약하게 연결
    cite_src, cite_dst = [], []
    for s in range(n_subject):
        members = torch.where(paper_subject == s)[0].tolist()
        for i in range(len(members) - 1):
            cite_src.append(members[i]); cite_dst.append(members[i + 1])
    cite = torch.tensor([cite_src, cite_dst], dtype=torch.long) if cite_src else torch.zeros((2, 0), dtype=torch.long)
    d["paper", "cite", "paper"].edge_index = cite
    d["paper", "ref", "paper"].edge_index = cite[[1, 0]] if cite.numel() else cite

    # train/val/test 마스크
    perm = torch.randperm(n_paper, generator=g)
    tr, va = int(0.5 * n_paper), int(0.7 * n_paper)
    train_mask = torch.zeros(n_paper, dtype=torch.bool); train_mask[perm[:tr]] = True
    val_mask = torch.zeros(n_paper, dtype=torch.bool); val_mask[perm[tr:va]] = True
    test_mask = torch.zeros(n_paper, dtype=torch.bool); test_mask[perm[va:]] = True
    d["paper"].train_mask = train_mask
    d["paper"].val_mask = val_mask
    d["paper"].test_mask = test_mask
    d["paper"].num_classes = n_class
    return d, "paper"
