#!/bin/bash
# GTN NaN 버그픽스 후 imdb·freebase 재실행: {c2_gtn, b2_mix, e2_mix, f_rgcn} × 10 seed = 80 task, ≤7 GPU.
# 완료 시 두 데이터셋 집계를 results/rerun_imdb_freebase.md 로만 push (SUMMARY 는 안 건드림 — 협업자 값 보호).
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=7
LOG="slurm_logs/rerun_nan_$(date +%Y%m%d_%H%M%S).log"
{
  echo "[rerun] $(date) start"
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  for ds in imdb freebase; do
    for st in c2_gtn b2_mix e2_mix f_rgcn; do
      echo "configs/campaign/${ds}__${st}.json"
    done
  done > configs/campaign/manifest.txt
  NSEED=$(grep -c . experiments/seeds.txt); N=$((8 * NSEED))
  sbatch --array=0-$((N - 1))%${MAXGPU} experiments/run_campaign.slurm \
    && echo "[rerun] submitted 0-$((N-1)) %${MAXGPU} (tasks=$N)"
  sleep 30
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  echo "[rerun] runs done -> aggregate"
  python - <<'PY'
import glob, json
import numpy as np
SEEDS = set(int(x) for x in open("experiments/seeds.txt").read().split())
L = ["# imdb·freebase 재실행 결과 (GTN NaN 버그픽스 후)\n",
     "이전 run 은 GTN NaN→위상 0벡터(silent failure)여서 무효 — `_gtn_norm` clamp 픽스 후 재실행.",
     "조건: c2_gtn=(c) HAN+위상, b2_mix=(b2), e2_mix=(e2), f_rgcn=(f). test Macro-F1 / accuracy, mean±std (10 seed).\n",
     "| 데이터셋 | 조건 | macro-F1 | accuracy | n | GTN-attn NaN run |",
     "|---|---|---|---|---|---|"]
for ds in ["imdb", "freebase"]:
    for st in ["c2_gtn", "b2_mix", "e2_mix", "f_rgcn"]:
        f1s, accs, nan_att, n = [], [], 0, 0
        for p in glob.glob(f"runs/campaign/{ds}__{st}_s*/metrics.json"):
            s = int(p.split("_s")[-1].split("/")[0])
            if s not in SEEDS: continue
            d = json.load(open(p)); n += 1
            v = d.get("test_macro_f1"); a = d.get("test_accuracy")
            if v is not None and np.isfinite(v): f1s.append(v)
            if a is not None and np.isfinite(a): accs.append(a)
            att = d.get("gtn_attentions")
            if att:
                flat = [x for layer in att for w in layer for x in np.ravel(np.array(w, float))]
                if np.isnan(flat).any(): nan_att += 1
        f1 = f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}" if f1s else "(미완)"
        ac = f"{np.mean(accs):.3f}±{np.std(accs):.3f}" if accs else "(미완)"
        L.append(f"| {ds} | {st} | {f1} | {ac} | {n} | {nan_att} |")
open("results/rerun_imdb_freebase.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
PY
  git add results/rerun_imdb_freebase.md
  git commit -q -m "rerun

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
  git pull --rebase -q origin main || true
  git push -q origin main && echo "[rerun] pushed" || echo "[rerun] push failed"
  echo "[rerun] done $(date)"
} >> "$LOG" 2>&1
echo "rerun logged -> $LOG"
