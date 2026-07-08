#!/usr/bin/env bash
# Unattended pipeline: wait for the HAN fine-tune sweep to finish, pick the best
# config by mean valid ROC-AUC, retrain it once with a saved checkpoint, then run
# a multi-seed test evaluation. Intended to be launched with nohup so the whole
# chain completes without a live terminal.
#
# Usage:
#   cd molhiv_tda && nohup bash scripts/run_finetune_best_pipeline.sh \
#       > logs/finetune_pipeline.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."  # -> molhiv_tda project root
CONFIG="${CONFIG:-pdgnn_han_finetune_3d_elec_multifilt}"
SWEEP_DIR="results/sweep/${CONFIG}_hantune"
SUMMARY="$SWEEP_DIR/_summary.json"
BACKBONE_CKPT="results/pdgnn_tda_3d_elec_best.pt"
EPOCHS="${EPOCHS:-30}"
CKPT_OUT="${CKPT_OUT:-results/${CONFIG}_best.pt}"
JSON_OUT="${JSON_OUT:-results/${CONFIG}_best.json}"
# Pass --multifilt to the evaluator when the config carries the A+B lenses.
MULTIFILT_FLAG=""
[[ "$CONFIG" == *multifilt* ]] && MULTIFILT_FLAG="--multifilt"

echo ">> [$(date -u +%H:%M:%S)] waiting for sweep to finish..."
while pgrep -f "[s]weep_pdgnn_han_finetune.py" >/dev/null 2>&1; do
  sleep 30
done
echo ">> [$(date -u +%H:%M:%S)] sweep done; reading best config from $SUMMARY"

python - "$SUMMARY" > /tmp/best_cfg.txt <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
b = rows[0]  # summary is pre-sorted by valid_mean (desc)
print(b["lr"], b["dropout"], b["han_hidden"], b["han_layers"],
      b["han_heads"], b["han_dropout"], int(b["balanced_train"]))
PY
read -r LR DO HH HL HD HDO BT < /tmp/best_cfg.txt
echo ">> best config: lr=$LR head_do=$DO han_h=$HH han_L=$HL heads=$HD han_do=$HDO balanced_train=$BT"

BAL=""
[ "$BT" = "1" ] && BAL="--balanced-train"

echo ">> [$(date -u +%H:%M:%S)] retraining best with --save-ckpt (epochs=$EPOCHS)..."
python -u train/train_pdgnn_han_finetune.py --config "$CONFIG" \
  --backbone-ckpt "$BACKBONE_CKPT" --lr "$LR" --dropout "$DO" \
  --han-hidden "$HH" --han-layers "$HL" --han-heads "$HD" --han-dropout "$HDO" \
  $BAL --epochs "$EPOCHS" --seed 0 --device cuda \
  --out "$JSON_OUT" \
  --save-ckpt "$CKPT_OUT"

echo ">> [$(date -u +%H:%M:%S)] multi-seed test evaluation of the fine-tune checkpoint..."
python -u scripts/eval_pdgnn_han_finetune_ckpt.py \
  --ckpt "$CKPT_OUT" $MULTIFILT_FLAG \
  --han-hidden "$HH" --han-layers "$HL" --han-heads "$HD" --han-dropout "$HDO" \
  --dropout "$DO" --balance-seeds 30

echo ">> [$(date -u +%H:%M:%S)] pipeline complete."
