#!/bin/bash
# 확장 캠페인 (신규 4 ds: imdb·freebase·mag·yelp):
#   ① 주입 factorial gt_*+gh_* (4 ds × 14 config × 10 seed = 560 task)
#   ② LP L2 lp2_* (4 ds × 4 config × 10 seed = 160 task)
# warm: 위상 캐시 4개(gated) + pair-EPD 캐시 4개(LP2) 병렬 → 본대 청크(≤90) %MAXGPU.
# 실행:  MAXGPU=16 bash experiments/run_ext_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-8}
NEW_DS="imdb freebase mag yelp"
GM=configs/campaign/gated_ext_manifest.txt
LM=configs/campaign/lp2_ext_manifest.txt
NSEED=10
head -${NSEED} experiments/seeds.txt > experiments/seeds_lp2.txt

python experiments/gen_gated.py >/dev/null
python experiments/gen_lp.py >/dev/null
: > "$GM"
for ds in $NEW_DS; do
  for c in gt_base gt_cat_real gt_cat_noise gt_cat_mix gt_gate_real gt_gate_noise gt_gate_mix \
           gh_base gh_cat_real gh_cat_noise gh_cat_mix gh_gate_real gh_gate_noise gh_gate_mix; do
    echo "configs/campaign/${ds}__${c}.json" >> "$GM"
  done
done
: > "$LM"
for ds in $NEW_DS; do
  for c in lp2_base lp2_real lp2_noise lp2_mix; do
    echo "configs/campaign/${ds}__${c}.json" >> "$LM"
  done
done
NG=$(( $(wc -l < "$GM") * NSEED ))
NL=$(( $(wc -l < "$LM") * NSEED ))
echo "gated ext=$NG, lp2 ext=$NL (동시 ≤$MAXGPU)"

# ---- warm (병렬 8개: gated 위상 4 + pair-EPD 4) ----
WG=""; for ds in $NEW_DS; do
  i=$(grep -n "${ds}__gt_gate_real.json" "$GM" | head -1 | cut -d: -f1)
  WG="$WG,$(( (i-1) * NSEED ))"
done
WL=""; for ds in $NEW_DS; do
  i=$(grep -n "${ds}__lp2_real.json" "$LM" | head -1 | cut -d: -f1)
  WL="$WL,$(( (i-1) * NSEED ))"
done
w1=$(sbatch --parsable --export=ALL,GATED_MANIFEST=$GM --array=${WG#,}%4 experiments/run_gated.slurm) \
  && echo "warm gated $w1 (${WG#,})" || echo "SBATCH FAILED warm gated"
w2=$(sbatch --parsable --export=ALL,LP_MANIFEST=$LM,LP_SEEDS=experiments/seeds_lp2.txt --array=${WL#,}%4 experiments/run_lp.slurm) \
  && echo "warm lp2 $w2 (${WL#,})" || echo "SBATCH FAILED warm lp2"
sleep 30
while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
echo "warm done -> mains"

# ---- 본대: gated 560 (청크 ≤90) → lp2 160 (청크 ≤90) ----
submit_chunks() {  # $1=total $2=slurm $3=extra-export
  local NEXT=0 CHUNK=90 END
  while [ $NEXT -lt $1 ]; do
    while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
    END=$((NEXT + CHUNK - 1)); [ $END -ge $1 ] && END=$(( $1 - 1 ))
    sbatch --export=$3 --array=${NEXT}-${END}%${MAXGPU} "$2" \
      && echo "submitted $2 ${NEXT}-${END} %${MAXGPU}" || echo "SBATCH FAILED $2 ${NEXT}-${END}"
    NEXT=$((END + 1))
  done
}
submit_chunks "$NG" experiments/run_gated.slurm "ALL,GATED_MANIFEST=$GM"
while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
submit_chunks "$NL" experiments/run_lp.slurm "ALL,LP_MANIFEST=$LM,LP_SEEDS=experiments/seeds_lp2.txt"
echo "all submitted"
