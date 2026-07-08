#!/usr/bin/env python3
"""Download OGBG-MolHIV into molhiv_tda/dataset (for fresh cloud instances)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MOLHIV_TDA = PROJECT_ROOT.parent
sys.path.insert(0, str(MOLHIV_TDA))

from ogb.graphproppred import PygGraphPropPredDataset  # noqa: E402

from data.load_molhiv import (  # noqa: E402
    DATASET_NAME,
    _patch_torch_load,
    _restore_torch_load,
    prepare_dataset,
)


def download_molhiv(dataset_root: Path | None = None) -> Path:
    root = dataset_root or (MOLHIV_TDA / "dataset")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_NAME} to {root} ...")
    _patch_torch_load()
    try:
        dataset = PygGraphPropPredDataset(name=DATASET_NAME, root=str(root))
    finally:
        _restore_torch_load()

    prepare_dataset(root)
    print(f"Ready: {len(dataset)} graphs at {root / 'ogbg_molhiv'}")
    return root


if __name__ == "__main__":
    download_molhiv()
