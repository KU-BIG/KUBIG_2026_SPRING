"""LP L1+L2 집계 → results/LP.md (test AUC/AP mean±std, L2 는 CN baseline 포함)."""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
DS = ["acm", "dblp", "aifb", "imdb", "freebase", "mag", "yelp"]
L1 = [("lp_a", "base"), ("lp_b1", "+noise"), ("lp_m", "+mix(node)"), ("lp_c", "+node-PI")]
L2 = [("lp2_base", "base"), ("lp2_noise", "+noise"), ("lp2_mix", "+mix(CN)"), ("lp2_real", "+pair-PI")]


def agg(ds, cond, key="test_auc"):
    xs = []
    for p in glob.glob(f"runs/lp/{ds}__{cond}_s*/metrics.json"):
        v = json.load(open(p)).get(key)
        if v is not None and np.isfinite(v):
            xs.append(v)
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)


def cell(ds, cond, key="test_auc"):
    m, s, n = agg(ds, cond, key)
    return f"{m:.3f}±{s:.3f} (n={n})" if n else "(미완)"


L = ["# LP 결과 — 위상 특징 × link prediction (test AUC, mean±std)\n",
     "설계: docs/lp_design.md. 예측 대상 = 주 타겟-타겟 관계 엣지, 고정 split, "
     "encoder=RGCN. L1=node-PI encoder concat(스캔, 5 seed) / L2=pair-vicinity EPD "
     "decoder concat(TLC-GNN 식, 10 seed).\n",
     "## Level 1 — node-PI (encoder 입력 concat)\n",
     "| 데이터셋 | " + " | ".join(l for _, l in L1) + " | +node-PI(관계제외 c') |",
     "|---|" + "---|" * (len(L1) + 1)]
for ds in DS:
    row = [ds] + [cell(ds, c) for c, _ in L1]
    row.append(cell(ds, "lp_cx") if ds == "dblp" else "–")
    L.append("| " + " | ".join(row) + " |")
L += ["", "## Level 2 — pair-vicinity EPD (decoder concat, TLC-GNN 식)\n",
      "| 데이터셋 | " + " | ".join(l for _, l in L2) + " | CN 휴리스틱 단독 |",
      "|---|" + "---|" * (len(L2) + 1)]
for ds in DS:
    row = [ds] + [cell(ds, c) for c, _ in L2]
    row.append(cell(ds, "lp2_real", "test_auc_cn"))
    L.append("| " + " | ".join(row) + " |")
L += ["", "판정: real>mix>noise ⇒ pair-위상 고유 신호 / real≈mix ⇒ CN 수준 정보뿐 / "
      "전부≈base ⇒ LP 에서도 null (AP 는 metrics.json 의 test_ap).", ""]
os.makedirs("results", exist_ok=True)
open("results/LP.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
