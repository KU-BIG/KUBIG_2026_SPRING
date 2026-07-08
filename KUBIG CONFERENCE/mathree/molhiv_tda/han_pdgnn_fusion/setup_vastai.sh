#!/usr/bin/env bash
# One-shot GPU setup for vast.ai (or any CUDA Linux box).
set -euo pipefail

FUSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FUSION_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"

# Pin versions that work on vast.ai (avoids pyg-lib / libpyg.so mismatch on cu128/cu126).
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
CUDA_TAG="${CUDA_TAG:-cu124}"

echo "Repo: $REPO_ROOT"
echo "Venv: $VENV"
echo "PyTorch: ${TORCH_VERSION}+${CUDA_TAG}"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools

echo "Removing old PyTorch / PyG wheels (if any)..."
pip uninstall -y pyg-lib torch-scatter torch-sparse torch-geometric torch torchvision 2>/dev/null || true

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "Installing PyTorch ${TORCH_VERSION} (${CUDA_TAG})..."
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX"

echo "Installing PyG extensions..."
pip install pyg-lib torch-scatter torch-sparse -f "$PYG_INDEX"
pip install torch-geometric
pip install -r "$FUSION_DIR/requirements.txt"

echo "Verifying CUDA + PyG..."
python - <<'PY'
import torch
import torch_scatter
import torch_sparse
from torch_geometric.utils import scatter

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0])
idx = torch.tensor([0, 0, 1])
scatter(x, idx, dim=0, dim_size=2, reduce="sum")
print("PyG OK")
PY

echo "Downloading OGBG-MolHIV (first run only)..."
python "$FUSION_DIR/download_molhiv.py"

echo ""
echo "Setup complete."
echo "  source $VENV/bin/activate"
echo "  cd $FUSION_DIR"
echo "  python -u train_han_pdgnn.py --model main --epochs 100 --device cuda"
