"""3D conformer generation and distance-based TDA features."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges

from config import PI_RESOLUTION, PI_SIGMA, RIPS_MAX_DIM, RIPS_MAX_EDGE, TDA_3D_DIM


def smiles_to_3d_points(smiles: str) -> Optional[np.ndarray]:
    """Generate 3D atom coordinates from SMILES using RDKit ETKDG (with hydrogens)."""
    return _embed_3d_points(smiles, add_hs=True)


def smiles_to_3d_points_heavy(smiles: str, num_nodes: int) -> Optional[np.ndarray]:
    """Generate 3D coordinates for heavy atoms only (matches OGB graph node count)."""
    return _embed_3d_points(smiles, add_hs=False, expected_atoms=num_nodes)


def _embed_3d_points(
    smiles: str,
    add_hs: bool,
    expected_atoms: int | None = None,
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if expected_atoms is not None and mol.GetNumAtoms() != expected_atoms:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    points = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        points.append([pos.x, pos.y, pos.z])
    return np.asarray(points, dtype=np.float64)


def edge_distances_from_points(
    edge_index: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Euclidean 3D distance for each edge in edge_index order."""
    dists = []
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        dists.append(float(np.linalg.norm(points[u] - points[v])))
    return np.asarray(dists, dtype=np.float32)


def compute_edge_distances_for_graph(
    smiles: str,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[np.ndarray, bool]:
    """Return per-edge 3D distances aligned with PyG edge_index."""
    points = smiles_to_3d_points_heavy(smiles, num_nodes)
    if points is None:
        return np.zeros(edge_index.size(1), dtype=np.float32), False
    ei = edge_index.cpu().numpy()
    return edge_distances_from_points(ei, points), True


def compute_edge_electrostatic_for_graph(
    smiles: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, bool]:
    """
    Return per-edge [distance, coulomb_like] aligned with edge_index.

    coulomb_like = (q_i * q_j) / (r_ij^2 + eps)
    where q are Gasteiger partial charges from RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != num_nodes:
        return np.zeros((edge_index.size(1), 2), dtype=np.float32), False

    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        return np.zeros((edge_index.size(1), 2), dtype=np.float32), False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        return np.zeros((edge_index.size(1), 2), dtype=np.float32), False

    conf = mol.GetConformer()
    charges = np.zeros(num_nodes, dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        val = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        try:
            charges[i] = float(val)
        except Exception:
            charges[i] = 0.0

    ei = edge_index.cpu().numpy()
    out = np.zeros((ei.shape[1], 2), dtype=np.float32)
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        pu = conf.GetAtomPosition(u)
        pv = conf.GetAtomPosition(v)
        r = float(np.linalg.norm([pu.x - pv.x, pu.y - pv.y, pu.z - pv.z]))
        coul = float((charges[u] * charges[v]) / (r * r + eps))
        out[e, 0] = r
        out[e, 1] = coul
    return out, True


def points_to_persistence_diagrams(
    points: np.ndarray,
    max_edge_length: float = RIPS_MAX_EDGE,
    max_dimension: int = RIPS_MAX_DIM,
) -> list[list]:
    """Compute Vietoris-Rips persistence diagrams from a 3D point cloud."""
    import gudhi

    from features.tda_utils import PersistencePoint

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence()

    diagrams: list[list[PersistencePoint]] = [
        [] for _ in range(max_dimension + 1)
    ]
    for dim, (birth, death) in st.persistence():
        if dim > max_dimension:
            continue
        if birth == float("inf") or death == float("inf"):
            continue
        if death <= birth:
            continue
        diagrams[dim].append(PersistencePoint(float(birth), float(death)))
    return diagrams


def compute_3d_tda_vector(
    smiles: str,
    resolution: int = PI_RESOLUTION,
    sigma: float = PI_SIGMA,
) -> tuple[np.ndarray, bool]:
    """
    Compute 3D distance-filtration TDA vector.
    Returns (vector, success_flag).
    """
    from features.tda_utils import persistence_image

    points = smiles_to_3d_points(smiles)
    if points is None or len(points) == 0:
        return np.zeros(TDA_3D_DIM, dtype=np.float32), False

    diagrams = points_to_persistence_diagrams(points)
    vectors = []
    for dim, diag in enumerate(diagrams):
        vectors.append(
            persistence_image(
                diag,
                resolution=resolution,
                sigma=sigma,
                birth_range=(0.0, RIPS_MAX_EDGE),
                pers_range=(0.0, RIPS_MAX_EDGE),
            )
        )
    vector = np.concatenate(vectors, axis=0).astype(np.float32)
    if vector.shape[0] < TDA_3D_DIM:
        vector = np.pad(vector, (0, TDA_3D_DIM - vector.shape[0]))
    elif vector.shape[0] > TDA_3D_DIM:
        vector = vector[:TDA_3D_DIM]
    return vector, True
