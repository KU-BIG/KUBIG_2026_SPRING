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

L = ["# TDA 실험 결과 (통합 캠페인)\n",
     "노드 분류 test Macro-F1, **mean±std over seed 0/1/2** (성능 주장 아닌 실측). 원본: `results/campaign/`.\n",
     "## 1. 여러 도메인: baseline(A1) vs full(C2)\n",
     "| 데이터셋 | 도메인 | A1 baseline | C2 full | Δmean |",
     "|----------|--------|-------------|---------|-------|"]
for ds in DATASETS:
    a = agg(ds, "a1_baseline"); c = agg(ds, "c2_gtn")
    dm = f"{c[0]-a[0]:+.4f}" if (a[2] and c[2]) else "—"
    L.append(f"| {ds} | {DOMAIN[ds]} | {cell(ds,'a1_baseline')} | {cell(ds,'c2_gtn')} | {dm} |")

L += ["", "## 2. 데이터셋별 A~D ablation (mean±std)\n",
      "| 데이터셋 | A3 GTN단독 | B2 manual | D2 noMIN | D3 ch2 | D3 ch8 | D4 L1 | D4 L3 | D5 random | topo-only |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    L.append("| " + " | ".join([ds,
        cell(ds, "c2_gtn", "gtn_only_test_macro_f1"), cell(ds, "b2_manual"),
        cell(ds, "d2_nomin"), cell(ds, "d3_ch2"), cell(ds, "d3_ch8"),
        cell(ds, "d4_l1"), cell(ds, "d4_l3"), cell(ds, "d5_random"),
        cell(ds, "topoonly")]) + " |")

L += ["", "## 3. 진단: permutation (위상 셔플 시 C2)\n",
      "| 데이터셋 | C2 | C2(permuted) | Δ |", "|---|---|---|---|"]
for ds in DATASETS:
    c = agg(ds, "c2_gtn"); p = agg(ds, "c2_gtn", "test_macro_f1_permuted")
    d = f"{c[0]-p[0]:+.4f}" if (c[2] and p[2]) else "—"
    L.append(f"| {ds} | {cell(ds,'c2_gtn')} | {cell(ds,'c2_gtn','test_macro_f1_permuted')} | {d} |")

L += ["", "주: D1(no-kNN)은 ego 폭증으로 비현실적(>2h)이라 제외 → kNN 필수.",
      f"진척: {len(C)} / {len(DATASETS)*10*3} run 완료.", ""]
open("results/SUMMARY.md", "w").write("\n".join(L) + "\n")
print(f"{len(C)} runs aggregated")
print("\n".join(L[:8]))
