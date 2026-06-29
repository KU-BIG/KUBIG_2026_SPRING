"""통합 캠페인(runs/campaign) 결과를 results/ 로 모으고 SUMMARY.md 재생성.
8 데이터셋 × A~D 설정 × seed 0/1/2 를 mean±std 로 집계. 부분 완료 상태도 안전하게 처리."""
import glob
import json
import os
import shutil

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
os.makedirs("results/campaign", exist_ok=True)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


for sub in ("runs/campaign", "runs/campaign_rdf"):
    for d in glob.glob(f"{sub}/*/"):
        n = os.path.basename(d.rstrip("/"))
        if os.path.exists(f"{d}/metrics.json"):
            shutil.copy(f"{d}/metrics.json", f"results/campaign/{n}.json")

C = {os.path.basename(p)[:-5]: load(p) for p in glob.glob("results/campaign/*.json")}


def agg(ds, setting, key="test_macro_f1"):
    xs = [C[f"{ds}__{setting}_s{s}"][key] for s in (0, 1, 2)
          if f"{ds}__{setting}_s{s}" in C and C[f"{ds}__{setting}_s{s}"]
          and C[f"{ds}__{setting}_s{s}"].get(key) is not None]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)


def cell(ds, setting, key="test_macro_f1"):
    m, s, n = agg(ds, setting, key)
    return f"{m:.4f}±{s:.4f}" if n else "—"


DATASETS = ["acm", "dblp", "dblp_pyg", "aminer", "mag", "imdb", "imdb_pyg", "freebase",
            "aifb", "mutag", "bgs", "am", "pubmed", "yelp"]
DOMAIN = {"acm": "학술/인용", "dblp": "학술/인용", "dblp_pyg": "학술(PyG판)",
          "aminer": "학술(대형·subsample·featureless)", "mag": "학술(대형·subsample)",
          "imdb": "영화(멀티라벨)", "imdb_pyg": "영화(단일라벨)", "freebase": "지식그래프(featureless)",
          "aifb": "RDF·연구기관", "mutag": "RDF·화학", "bgs": "RDF·지질", "am": "RDF·박물관",
          "pubmed": "생의학(HNE)", "yelp": "business(HNE·멀티라벨)"}
# 캠페인에 아직 안 들어온(결과 없는) 데이터셋은 표에서 자동 생략

DATASETS = [ds for ds in DATASETS if any(k.startswith(ds + "__") for k in C)]  # 결과 있는 것만

def cm(ds, st, key="test_macro_f1"):   # compact mean±std
    m, s, n = agg(ds, st, key)
    return f"{m:.3f}±{s:.2f}" if n else "—"

# 모든 A~D 실험(+진단)을 하나의 표로. test Macro-F1, mean±std over seed 0/1/2.
L = ["# TDA 실험 결과 — 14개 데이터셋 × A~D 통합표\n",
     "노드 분류 **test Macro-F1, mean±std (seed 0/1/2)**. 성능 주장 아닌 실측. 원본: `results/campaign/`.",
     "열: A1=HAN단독, A3=GTN단독, B2=manual메타패스+EPD, **C2=GTN+PDGNN+HAN(메인)**, "
     "D2=MIN제거, D3=채널2/8, D4=깊이1/3, D5=random메타패스, topo=위상만, perm=위상셔플, "
     "Δ=C2−A1. (D1=no-kNN 은 비현실적(>2h)이라 제외.)\n",
     "| 데이터셋 | 도메인 | A1 | A3 | B2 | **C2** | D2 | D3=2 | D3=8 | D4=1 | D4=3 | D5 | topo | perm | Δ(C2−A1) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    a1 = agg(ds, "a1_baseline"); c2 = agg(ds, "c2_gtn")
    dlt = f"**{c2[0]-a1[0]:+.3f}**" if (a1[2] and c2[2]) else "—"
    L.append("| " + " | ".join([
        ds, DOMAIN[ds],
        cm(ds, "a1_baseline"), cm(ds, "c2_gtn", "gtn_only_test_macro_f1"),
        cm(ds, "b2_manual"), cm(ds, "c2_gtn"), cm(ds, "d2_nomin"),
        cm(ds, "d3_ch2"), cm(ds, "d3_ch8"), cm(ds, "d4_l1"), cm(ds, "d4_l3"),
        cm(ds, "d5_random"), cm(ds, "topoonly"),
        cm(ds, "c2_gtn", "test_macro_f1_permuted"), dlt]) + " |")

L += ["", f"진척: {len(C)} / {len(DATASETS)*10*3} run.",
      "해석: 위상(C2)의 효용은 node feature 가 약/없을 때 큼(DBLP·AMiner·AIFB 등 Δ↑), "
      "feature 가 강하면 ≈0(ACM·dblp_pyg·IMDB). MUTAG/BGS/AM 은 featureless RDF 라 degenerate.", ""]
open("results/SUMMARY.md", "w").write("\n".join(L) + "\n")
print(f"{len(C)} runs aggregated")
print("\n".join(L[:8]))
