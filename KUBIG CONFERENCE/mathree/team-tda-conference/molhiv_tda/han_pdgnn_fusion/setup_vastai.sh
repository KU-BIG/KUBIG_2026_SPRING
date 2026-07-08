#!/usr/bin/env bash
# One-shot GPU setup for vast.ai (or any CUDA Linux box).
set -euo pipefail

FUSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FUSION_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"

echo "Repo: $REPO_ROOT"
echo "Venv: $VENV"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools

# Pick CUDA wheel tag from driver (default cu126).
CUDA_TAG="${CUDA_TAG:-cu126}"
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
  case "$DRIVER_CUDA" in
    12.4|12.5) CUDA_TAG=cu124 ;;
    12.6|12.7|12.8) CUDA_TAG=cu126 ;;
    12.9|13.*) CUDA_TAG=cu128 ;;
  esac
  echo "Detected driver CUDA $DRIVER_CUDA -> PyTorch tag $CUDA_TAG"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-2.12.0+${CUDA_TAG}.html"

pip install torch torchvision --index-url "$TORCH_INDEX"
pip install pyg-lib torch-scatter torch-sparse -f "$PYG_INDEX"
pip install -r "$FUSION_DIR/requirements.txt"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Downloading OGBG-MolHIV (first run only)..."
python "$FUSION_DIR/download_molhiv.py"

echo ""
echo "Setup complete."
echo "  source $VENV/bin/activate"
echo "  cd $FUSION_DIR"
echo "  python train_han_pdgnn.py --model main --epochs 100 --device cuda"
