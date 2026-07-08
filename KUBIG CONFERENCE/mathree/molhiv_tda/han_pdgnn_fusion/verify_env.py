#!/usr/bin/env python3
"""Quick environment check before training on vast.ai."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MOLHIV_TDA = PROJECT_ROOT.parent
sys.path.insert(0, str(MOLHIV_TDA))
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    errors: list[str] = []

    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except Exception as exc:
        errors.append(f"torch: {exc}")

    try:
        import torch
        import torch_scatter
        import torch_sparse
        from torch_geometric.utils import scatter

        scatter(torch.tensor([1.0, 2.0]), torch.tensor([0, 1]), dim=0, dim_size=2)
        print("PyG: OK")
    except Exception as exc:
        errors.append(f"PyG: {exc} (re-run: CUDA_TAG=cu124 bash setup_vastai.sh)")

    dataset_root = MOLHIV_TDA / "dataset"
    hiv_csv = dataset_root / "ogbg_molhiv" / "mapping" / "hiv.csv"
    processed = dataset_root / "ogbg_molhiv" / "processed" / "geometric_data_processed.pt"
    if not processed.exists():
        errors.append(f"Missing dataset: {processed} (run: python download_molhiv.py)")
    if not hiv_csv.exists():
        errors.append(f"Missing SMILES map: {hiv_csv} (run: python download_molhiv.py)")

    if errors:
        print("\nEnvironment check FAILED:")
        for msg in errors:
            print(" -", msg)
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
