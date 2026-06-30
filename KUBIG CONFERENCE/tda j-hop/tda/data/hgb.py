"""HGB benchmark 이종 NC 데이터셋 로더 (DBLP/IMDB/Freebase). PyG `HGBDataset`.

ACM 과 동일하게, val 마스크가 없으면 train 의 15% 를 시드 고정 무작위로 분리한다.
Freebase 는 노드 특징이 없어 registry 에서 one-hot 으로 대체된다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _load_hgb(name: str, target: str, data_root: str) -> Tuple[object, str]:
    from torch_geometric.datasets import HGBDataset

    d = HGBDataset(root=f"{data_root}/HGB_{name}", name=name)[0]
    p = d[target]
    if not hasattr(p, "val_mask") or p.val_mask is None:
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


def load_dblp(data_root: str = "./data"):
    return _load_hgb("DBLP", "author", data_root)


def load_imdb(data_root: str = "./data"):
    return _load_hgb("IMDB", "movie", data_root)


def load_freebase(data_root: str = "./data"):
    return _load_hgb("Freebase", "book", data_root)


def load_dblp_pyg(data_root: str = "./data"):
    """PyG 단독 DBLP (HAN/MAGNN 판, HGB DBLP 와 다른 전처리). 마스크 내장, author 4클래스."""
    from torch_geometric.datasets import DBLP

    return DBLP(root=f"{data_root}/DBLP_pyg")[0], "author"


def load_imdb_pyg(data_root: str = "./data"):
    """PyG 단독 IMDB (MAGNN 판). 마스크 내장, movie 3클래스 **단일라벨**(HGB IMDB 멀티라벨과 다름)."""
    from torch_geometric.datasets import IMDB

    return IMDB(root=f"{data_root}/IMDB_pyg")[0], "movie"
