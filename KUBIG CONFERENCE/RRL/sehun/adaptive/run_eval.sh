#!/usr/bin/env bash
source /workspace/venvs/cuda-test/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export PYTHONPATH=/workspace/rl_project/multiHRI
cd /workspace/rl_project/multiHRI/adaptive
mkdir -p data
python eval_adaptive.py --episodes 40 > data/eval.log 2>&1
echo "EXIT_$?" >> data/eval.log
