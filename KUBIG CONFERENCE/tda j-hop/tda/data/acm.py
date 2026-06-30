"""ACM 데이터셋 로더 (HGB benchmark, PyG `HGBDataset`).

타겟 = paper (3 클래스, Macro-F1). 노드 타입 paper/author/subject/term,
엣지 타입 paper-cite/ref-paper, paper-to-author, paper-to-subject, paper-to-term 등.

HGB ACM 에는 val 마스크가 없으므로 train 의 뒤쪽 15% 를 val 로 분리한다
(TLC-GNN `metapath_graph.build_metapath_graph` 와 동일한 방식).

데이터가 `data_root` 에 없으면 PyG 가 자동 다운로드한다. 이미 캐시가 있으면 그대로 사용.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def load_acm(data_root: str = "./data") -> Tuple[object, str]:
    """반환: (PyG HeteroData, target_node_type). val 마스크가 없으면 합성."""
    from torch_geometric.datasets import HGBDataset

    d = HGBDataset(root=f"{data_root}/HGB_ACM", name="ACM")[0]
    target = "paper"
    p = d[target]
    if not hasattr(p, "val_mask") or p.val_mask is None:
        # train 의 15% 를 val 로 분리. tail 이 아니라 시드 고정 무작위 분할을 쓴다
        # (HGB train 인덱스가 클래스 순으로 정렬돼 있어 tail 분할은 클래스 편향 -> 무의미한
        #  early stopping 을 유발한다).
        tr = p.train_mask.numpy().copy()
        idx = np.where(tr)[0]
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        cut = idx[int(0.85 * len(idx)):]
        val = np.zeros_like(tr)
        val[cut] = True
        tr[cut] = False
        p.train_mask = torch.from_numpy(tr)
        p.val_mask = torch.from_numpy(val)
    return d, target
