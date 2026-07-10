#!/usr/bin/env bash

source /workspace/venvs/cuda-test/bin/activate

export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/workspace/rl_project/multiHRI:${PYTHONPATH:-}

cd /workspace/rl_project/multiHRI

echo "Activated multiHRI environment"
echo "python: $(which python)"

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY
