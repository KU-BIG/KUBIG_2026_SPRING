"""실험 ①(주입×내용 factorial) 집계 → results/GATED.md."""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
SEEDS = set(int(x) for x in open("experiments/seeds.txt").read().split())
DS = ["acm", "dblp", "aifb", "imdb", "freebase", "mag", "yelp"]
COND = ["gt_base", "gt_cat_real", "gt_cat_noise", "gt_cat_mix",
        "gt_gate_real", "gt_gate_noise", "gt_gate_mix"]
LAB = {"gt_base": "base", "gt_cat_real": "concat+real", "gt_cat_noise": "concat+noise",
       "gt_cat_mix": "concat+mix", "gt_gate_real": "gate+real",
       "gt_gate_noise": "gate+noise", "gt_gate_mix": "gate+mix"}


def agg(ds, cond, key="test_macro_f1"):
    xs = []
    for p in glob.glob(f"runs/gated/{ds}__{cond}_s*/metrics.json"):
        s = int(p.split("_s")[-1].split("/")[0])
        if s not in SEEDS:
            continue
        v = json.load(open(p)).get(key)
        if v is not None and np.isfinite(v):
            xs.append(v)
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)


L = ["# 실험 ① — 주입(concat vs gate) × 내용(real/noise/mix) factorial (RGCN, NC)\n",
     "질문: (f)<(d) 의 위상 유해가 **concat 주입 방식** 탓인가? 위상 = manual 채널·topo_seed 고정"
     "(조건 간 동일 특징 — GTN/per-seed 아님). 모든 조건 동일 GatedRGCN 구현. test Macro-F1, mean±std.\n",
     "| 데이터셋 | " + " | ".join(LAB[c] for c in COND) + " |",
     "|---|" + "---|" * len(COND)]
for ds in DS:
    row = [ds]
    for c in COND:
        m, s, n = agg(ds, c)
        row.append(f"{m:.3f}±{s:.3f} (n={n})" if n else "(미완)")
    L.append("| " + " | ".join(row) + " |")
# HAN 백본 판 (gh_*)
COND_H = [c.replace("gt_", "gh_") for c in COND]
L += ["", "## HAN 백본 (동일 factorial, gh_*)\n",
      "| 데이터셋 | " + " | ".join(LAB[c] for c in COND) + " |",
      "|---|" + "---|" * len(COND)]
for ds in DS:
    row = [ds]
    for c in COND_H:
        m, s_, n = agg(ds, c)
        row.append(f"{m:.3f}\u00b1{s_:.3f} (n={n})" if n else "(\ubbf8\uc644)")
    L.append("| " + " | ".join(row) + " |")
L += ["", "판정 가이드: cat_real<base 이면서 gate_real≥base ⇒ 하락은 주입(concat) 탓. "
      "gate_real>gate_mix ⇒ 게이팅에선 per-node 정렬 기여 존재.", ""]
open("results/GATED.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
