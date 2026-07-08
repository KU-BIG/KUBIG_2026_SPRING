"""Partial-charge sublevel-set filtration TDA features (lens D).

Each atom gets a Gasteiger partial charge (RDKit, 2D, no conformer needed).
We build a lower-star (sublevel-set) filtration on the molecular graph: a node
enters the filtration at its charge value, an edge enters at max(charge_u,
charge_v). H0 then tracks how electronegative (low-charge) basins are born and
merge as the charge threshold sweeps upward; H1 tracks rings by the charge at
which they close.

This is rotation/translation invariant (Gasteiger charges are a function of the
2D graph only). It captures the *topology of the charge landscape* -- a cheap,
3D-free proxy for where a molecule concentrates charge -- but note that Gasteiger
charge is a deterministic function of the 2D graph the GNN already sees, so it
is an explicit inductive bias rather than genuinely new information. True
through-space electrostatics (q_i q_j / r_ij over non-bonded pairs) needs 3D.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from config import (
    CHARGE_BIRTH_RANGE,
    CHARGE_PERS_RANGE,
    CHARGE_PI_SIGMA,
    CHARGE_TDA_DIM,
    PI_RESOLUTION,
)
from features.tda_utils import gudhi_persistence_from_simplices, persistence_image


def smiles_to_gasteiger_charges(smiles: str) -> Optional[np.ndarray]:
    """Per-atom Gasteiger charges in RDKit atom order (== OGB node order).

    Returns None if the SMILES fails to parse. NaN/inf charges (Gasteiger can
    diverge on some atoms) are replaced with 0.0.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array(
        [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()],
        dtype=np.float64,
    )
    charges[~np.isfinite(charges)] = 0.0
    return charges


def _charge_simplices(data: Data, charges: np.ndarray) -> list[tuple[list[int], float]]:
    """Lower-star simplices: node at its charge, edge at max of endpoints."""
    num_nodes = int(data.num_nodes)
    simplices: list[tuple[list[int], float]] = []
    for node in range(num_nodes):
        c = float(charges[node]) if node < len(charges) else 0.0
        simplices.append(([node], c))

    ei = data.edge_index.cpu().numpy()
    seen: set[tuple[int, int]] = set()
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        cu = float(charges[u]) if u < len(charges) else 0.0
        cv = float(charges[v]) if v < len(charges) else 0.0
        simplices.append(([u, v], max(cu, cv)))

    simplices.sort(key=lambda x: x[1])
    return simplices


def compute_charge_tda_vector(data: Data, charges: Optional[np.ndarray]) -> np.ndarray:
    """Persistence-image vector (H0 + H1) for the charge sublevel filtration."""
    if charges is None or data.num_nodes == 0:
        return np.zeros(CHARGE_TDA_DIM, dtype=np.float32)

    simplices = _charge_simplices(data, charges)
    diagrams = gudhi_persistence_from_simplices(simplices, max_dim=1)

    vectors = [
        persistence_image(
            diag,
            resolution=PI_RESOLUTION,
            sigma=CHARGE_PI_SIGMA,
            birth_range=CHARGE_BIRTH_RANGE,
            pers_range=CHARGE_PERS_RANGE,
        )
        for diag in diagrams
    ]
    vector = np.concatenate(vectors, axis=0).astype(np.float32)

    if vector.shape[0] < CHARGE_TDA_DIM:
        vector = np.pad(vector, (0, CHARGE_TDA_DIM - vector.shape[0]))
    elif vector.shape[0] > CHARGE_TDA_DIM:
        vector = vector[:CHARGE_TDA_DIM]
    return vector


def compute_charge_tda_batch(
    dataset,
    smiles_list: list[str],
    indices: Optional[list[int]] = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, int]:
    """Compute charge-filtration vectors for selected graphs.

    Returns (features, num_failures) where failures are SMILES that would not parse.
    """
    if indices is None:
        indices = list(range(len(dataset)))

    iterator = indices
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(indices, desc="Charge TDA (lens D)")

    features = []
    failures = 0
    for idx in iterator:
        charges = smiles_to_gasteiger_charges(smiles_list[idx])
        if charges is None:
            failures += 1
        features.append(compute_charge_tda_vector(dataset[idx], charges))
    return torch.tensor(np.stack(features, axis=0), dtype=torch.float32), failures
