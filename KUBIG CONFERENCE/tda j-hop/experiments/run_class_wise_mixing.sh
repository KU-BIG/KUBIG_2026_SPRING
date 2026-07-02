#!/bin/bash
# Class-wise topology mixing ablation runner.
# Generates configs for 7 datasets x {HAN,RGCN}, then runs each config for seeds in experiments/seeds.txt.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data}
MAX_SEEDS=${MAX_SEEDS:-}

python experiments/gen_class_wise_mixing.py
mapfile -t CFGS < configs/campaign/class_wise_mixing_manifest.txt
mapfile -t SEEDS < experiments/seeds.txt
if [ -n "$MAX_SEEDS" ]; then
  SEEDS=("${SEEDS[@]:0:$MAX_SEEDS}")
fi

for CFG in "${CFGS[@]}"; do
  DS=$(python -c "import json;print(json.load(open('$CFG'))['dataset'])")
  BACKBONE=$(python -c "import json;print(json.load(open('$CFG')).get('backbone','han'))")
  MODE=$(python -c "import json;print(json.load(open('$CFG')).get('topology_mode','original'))")
  for SEED in "${SEEDS[@]}"; do
    OUT="runs/${MODE}/${DS}__${BACKBONE}__${MODE}_s${SEED}"
    LOG="${OUT}/run.log"
    if [ -f "$OUT/metrics.json" ]; then
      echo "skip existing $OUT"
      continue
    fi
    mkdir -p "$OUT"
    if [ -f "$LOG" ]; then
      mv "$LOG" "${LOG}.$(date +%Y%m%d_%H%M%S).partial"
    fi
    echo "[run_class_wise_mixing] ds=$DS backbone=$BACKBONE mode=$MODE seed=$SEED out=$OUT"
    python -m tda.train --config "$CFG" --dataset "$DS"       --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED" 2>&1 | tee "$LOG"
  done
done
