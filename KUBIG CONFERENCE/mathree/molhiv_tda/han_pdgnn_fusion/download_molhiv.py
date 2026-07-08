#!/usr/bin/env python3
"""Download OGBG-MolHIV into molhiv_tda/dataset (for fresh cloud instances)."""
from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

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

HIV_ZIP_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"


def ensure_smiles_mapping(dataset_root: Path) -> Path:
    """Ensure mapping/hiv.csv exists (needed for SMILES-based preprocessing)."""
    mapping_dir = dataset_root / "ogbg_molhiv" / "mapping"
    hiv_csv = mapping_dir / "hiv.csv"
    if hiv_csv.exists():
        return hiv_csv

    mapping_dir.mkdir(parents=True, exist_ok=True)

    mol_gz = mapping_dir / "mol.csv.gz"
    if mol_gz.exists():
        with gzip.open(mol_gz, "rb") as fin, open(hiv_csv, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"Created {hiv_csv} from mol.csv.gz")
        return hiv_csv

    mol_csv = mapping_dir / "mol.csv"
    if mol_csv.exists():
        shutil.copy(mol_csv, hiv_csv)
        print(f"Created {hiv_csv} from mol.csv")
        return hiv_csv

    print(f"Downloading SMILES mapping from {HIV_ZIP_URL} ...")
    with TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "hiv.zip"
        urllib.request.urlretrieve(HIV_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for candidate in Path(tmp).rglob("*.csv"):
            if candidate.name.lower() in {"hiv.csv", "mol.csv"}:
                shutil.copy(candidate, hiv_csv)
                print(f"Saved SMILES mapping to {hiv_csv}")
                return hiv_csv

    raise FileNotFoundError(
        f"Could not create {hiv_csv}. "
        "Run download_molhiv.py again or copy mapping/hiv.csv from a local machine."
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
    ensure_smiles_mapping(root)
    print(f"Ready: {len(dataset)} graphs at {root / 'ogbg_molhiv'}")
    return root


if __name__ == "__main__":
    download_molhiv()
