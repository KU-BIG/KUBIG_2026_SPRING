"""Molecular weight computation and normalization."""
from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


def smiles_to_molecular_weight(smiles: str) -> Optional[float]:
    """Return exact molecular weight from SMILES, or None if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.ExactMolWt(mol))


def compute_molecular_weights(smiles_list: list[str]) -> tuple[torch.Tensor, int]:
    """
    Compute molecular weights for all molecules.
    Returns (weights, num_failures).
    """
    weights = []
    failures = 0
    for smiles in smiles_list:
        mw = smiles_to_molecular_weight(smiles)
        if mw is None:
            weights.append(0.0)
            failures += 1
        else:
            weights.append(mw)
    return torch.tensor(weights, dtype=torch.float32).unsqueeze(1), failures
