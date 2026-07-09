#!/usr/bin/env bash
source /workspace/venvs/cuda-test/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export PYTHONPATH=/workspace/rl_project/multiHRI
cd /workspace/rl_project/multiHRI/adaptive
mkdir -p data
python gen_data.py --per_class 250 > data/gen.log 2>&1
echo "EXIT_$?" >> data/gen.log
