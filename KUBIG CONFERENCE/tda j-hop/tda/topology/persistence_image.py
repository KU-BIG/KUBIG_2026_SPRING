"""Persistence Image (Adams et al., JMLR 2017) — persistence diagram -> 고정 길이 벡터.

원본 TLC-GNN 파이프라인은 컴파일된 cython `sg2dgm.PersistenceImager`(.so)를 썼지만,
공유/Colab 호환을 위해 동일한 표준 PI 를 순수 numpy 로 재구현한다.

절차 (표준 PI):
  1. (birth, death) -> (birth, persistence=death-birth) 좌표 변환.
  2. 각 점에 가우시안 커널을 얹고, persistence 에 비례하는 선형 가중치 w(p)=clip(p)/p_max 를 곱함
     (수명이 긴 위상 특징일수록 가중).
  3. resolution x resolution 격자에서 적분(격자점 평가) 후 1차원으로 flatten.

birth/persistence 범위는 노드 간 비교 가능하도록 **고정**한다(기본값은 z-정규화된 HKS
필터에서 나오는 sublevel EPD 값 범위에 맞춤). 범위는 설정으로 바꿀 수 있다.
"""
from __future__ import annotations

import numpy as np


class PersistenceImage:
    def __init__(
        self,
        resolution: int = 5,
        sigma: float = 0.5,
        birth_range: tuple = (-3.0, 3.0),
        pers_range: tuple = (0.0, 6.0),
    ):
        self.resolution = int(resolution)
        self.sigma = float(sigma)
        self.birth_range = birth_range
        self.pers_range = pers_range
        # 격자 중심점 좌표 (resolution 개씩).
        self.bx = np.linspace(birth_range[0], birth_range[1], self.resolution)
        self.py = np.linspace(pers_range[0], pers_range[1], self.resolution)

    @property
    def dim(self) -> int:
        return self.resolution * self.resolution

    def transform(self, pairs: np.ndarray) -> np.ndarray:
        """pairs: (P, 2) = (birth, death). 반환: (resolution^2,) 벡터."""
        out = np.zeros(self.dim, dtype=np.float64)
        if pairs is None or len(pairs) == 0:
            return out
        pairs = np.asarray(pairs, dtype=np.float64)
        births = pairs[:, 0]
        pers = pairs[:, 1] - pairs[:, 0]
        keep = pers > 0
        if not np.any(keep):
            return out
        births, pers = births[keep], pers[keep]

        pmax = self.pers_range[1] if self.pers_range[1] > 0 else 1.0
        weights = np.clip(pers, 0.0, pmax) / pmax  # 선형 ramp 가중치 (0~1)

        # 격자점 (R, R): gx[i,j]=birth_j, gy[i,j]=pers_i
        gx, gy = np.meshgrid(self.bx, self.py)  # (R, R)
        gx = gx.reshape(-1)  # (R^2,)
        gy = gy.reshape(-1)
        inv2s2 = 1.0 / (2.0 * self.sigma * self.sigma)
        norm = 1.0 / (2.0 * np.pi * self.sigma * self.sigma)
        for b, p, w in zip(births, pers, weights):
            db = gx - b
            dp = gy - p
            g = norm * np.exp(-(db * db + dp * dp) * inv2s2)
            out += w * g
        return out
