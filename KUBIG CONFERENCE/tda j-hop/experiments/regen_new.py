"""새 설계 (a)~(f) 결과 정리 — 선정 7개 데이터셋만.
현재 실측: (a) HAN-only = a1_baseline, (c) HAN+GTN-PDGNN(learned) = c2_gtn.
미실행: (b) HAN+noisy(정의 미정), (d) RGCN-only, (e) RGCN+noisy, (f) RGCN+GTN-PDGNN (RGCN 미구현).
결과 -> results/SUMMARY.md"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))

DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
DOMAIN = {"acm": "학술/인용", "dblp": "학술/인용", "imdb": "영화(멀티라벨)",
          "freebase": "지식그래프", "mag": "학술(초대형)", "aifb": "RDF/연구기관",
          "yelp": "business(멀티라벨)"}
try:
    SEEDS = set(int(x) for x in open("experiments/seeds.txt").read().split())
except Exception:
    SEEDS = None
KEYS = ["test_macro_f1", "test_multilabel_macro_f1", "multilabel_macro_f1", "test_f1"]


def _metric(d):
    for k in KEYS:
        if d.get(k) is not None:
            return d[k]
    return None


def per_seed(ds, setting):
    out = {}
    for p in glob.glob(f"runs/campaign/{ds}__{setting}_s*/metrics.json"):
        try:
            s = int(p.split("_s")[-1].split("/")[0])
        except ValueError:
            continue
        if SEEDS is None or s in SEEDS:
            try:
                v = _metric(json.load(open(p)))
            except Exception:
                v = None
            if v is not None:
                out[s] = v
    return out


def cell(ds, setting):
    xs = list(per_seed(ds, setting).values())
    return f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if xs else "(미완)"


def nn(ds, setting):
    return len(per_seed(ds, setting))


seedline = " ".join(sorted((str(x) for x in SEEDS), key=int)) if SEEDS else "(seeds.txt 없음)"
L = ["# 새 실험 결과 — 7개 데이터셋 × (a)~(f)\n",
     f"노드 분류 **test Macro-F1, mean±std** (random seed 10개: {seedline}).\n",
     "(a)=HAN-only, (c)=HAN+GTN-PDGNN, (d)=RGCN-only, (f)=RGCN+GTN-PDGNN. "
     "(b)(e) noisy-topology 는 미실행.\n",
     "| 데이터셋 | 도메인 | (a) HAN | (b) HAN+noisy | (c) HAN+GTN-PDGNN | (d) RGCN | (e) RGCN+noisy | (f) RGCN+GTN-PDGNN |",
     "|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    a, c, d, f = nn(ds, "a1_baseline"), nn(ds, "c2_gtn"), nn(ds, "d_rgcn"), nn(ds, "f_rgcn")
    L.append("| " + " | ".join([
        ds, DOMAIN[ds],
        f"{cell(ds, 'a1_baseline')} (n={a})", "—(미실행)",
        f"{cell(ds, 'c2_gtn')} (n={c})",
        f"{cell(ds, 'd_rgcn')} (n={d})", "—(미실행)",
        f"{cell(ds, 'f_rgcn')} (n={f})"]) + " |")

da = sum(1 for ds in DATASETS if nn(ds, "a1_baseline") > 0)
dc = sum(1 for ds in DATASETS if nn(ds, "c2_gtn") > 0)
dd = sum(1 for ds in DATASETS if nn(ds, "d_rgcn") > 0)
df_ = sum(1 for ds in DATASETS if nn(ds, "f_rgcn") > 0)
L += ["",
      f"진척: **(a) {da}/7, (c) {dc}/7, (d) {dd}/7, (f) {df_}/7** 데이터셋 완료 (각 최대 10 seed). "
      "(b)(e) noisy-topology 미실행.",
      "",
      "## 매핑 / 상태",
      "- (a) HAN only = `a1_baseline`  ·  (c) HAN+GTN-PDGNN = `c2_gtn`  (backbone=han)",
      "- (d) RGCN only = `d_rgcn`  ·  (f) RGCN+GTN-PDGNN = `f_rgcn`  (backbone=rgcn, RGCNConv)",
      "- (b)(e) noisy topological → 정의 확정 필요 (미실행)",
      "",
      "원본 per-run: `runs/campaign/<ds>__{a1_baseline,c2_gtn}_s<seed>/metrics.json`", ""]
open("results/SUMMARY.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
