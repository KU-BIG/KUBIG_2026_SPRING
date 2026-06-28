"""공용 유틸: 시드 고정, 디바이스, 평가지표(Macro-F1), JSON 입출력, 로깅."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 RNG 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """다중 클래스 Macro-F1 (sklearn 의존 없이 직접 계산)."""
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    true = labels.detach().cpu().numpy()
    classes = np.unique(true)
    f1s = []
    for c in classes:
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Timer:
    """간단한 구간 타이머 (with 블록)."""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        if self.label:
            print(f"[time] {self.label}: {self.elapsed:.2f}s", flush=True)
