"""Persistence diagram utilities and persistence image vectorization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import gudhi
import numpy as np


@dataclass
class PersistencePoint:
    birth: float
    death: float


def gudhi_persistence_from_simplices(
    simplices: Sequence[Tuple[Sequence[int], float]],
    max_dim: int = 1,
) -> List[List[PersistencePoint]]:
    """Compute persistence diagrams from a list of (simplex, filtration) pairs."""
    st = gudhi.SimplexTree()
    for simplex, val in simplices:
        st.insert(list(simplex), filtration=float(val))
    st.compute_persistence()

    diagrams: List[List[PersistencePoint]] = [[] for _ in range(max_dim + 1)]
    for dim, (birth, death) in st.persistence():
        if dim > max_dim:
            continue
        if birth == float("inf") or death == float("inf"):
            continue
        if death <= birth:
            continue
        diagrams[dim].append(PersistencePoint(float(birth), float(death)))
    return diagrams


def persistence_image(
    points: Iterable[PersistencePoint],
    resolution: int = 5,
    sigma: float = 0.05,
    birth_range: Tuple[float, float] = (0.0, 4.0),
    pers_range: Tuple[float, float] = (0.0, 4.0),
) -> np.ndarray:
    """
    Convert a persistence diagram to a fixed-size persistence image vector.
    Uses birth-persistence coordinates with Gaussian weighting.
    """
    pts = list(points)
    grid = np.zeros((resolution, resolution), dtype=np.float64)
    if len(pts) == 0:
        return grid.reshape(-1)

    b_min, b_max = birth_range
    p_min, p_max = pers_range
    b_step = (b_max - b_min) / resolution
    p_step = (p_max - p_min) / resolution

    b_coords = b_min + (np.arange(resolution) + 0.5) * b_step
    p_coords = p_min + (np.arange(resolution) + 0.5) * p_step
    bb, pp = np.meshgrid(b_coords, p_coords, indexing="ij")

    for pt in pts:
        pers = pt.death - pt.birth
        if pers <= 0:
            continue
        dist2 = (bb - pt.birth) ** 2 + (pp - pers) ** 2
        grid += np.exp(-dist2 / (2.0 * sigma**2))

    return grid.reshape(-1)


def diagrams_to_vector(
    diagrams: Sequence[List[PersistencePoint]],
    resolution: int = 5,
    sigma: float = 0.05,
) -> np.ndarray:
    """Concatenate persistence images for each homology dimension."""
    vectors = [
        persistence_image(diag, resolution=resolution, sigma=sigma)
        for diag in diagrams
    ]
    return np.concatenate(vectors, axis=0).astype(np.float32)
