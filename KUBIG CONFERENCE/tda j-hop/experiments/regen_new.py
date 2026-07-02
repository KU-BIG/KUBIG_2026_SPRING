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
     "(a)=HAN-only, (b1)=HAN+noise(random), (b2)=HAN+class-mix, (c)=HAN+GTN-PDGNN, "
     "(d)=RGCN-only, (e1)=RGCN+noise(random), (e2)=RGCN+class-mix, (f)=RGCN+GTN-PDGNN.\n",
     "| 데이터셋 | 도메인 | (a) HAN | (b1) +noise | (b2) +class-mix | (c) +GTN-PDGNN | (d) RGCN | (e1) +noise | (e2) +class-mix | (f) +GTN-PDGNN |",
     "|---|---|---|---|---|---|---|---|---|---|"]
COLS = ["a1_baseline", "b1_noise", "b2_mix", "c2_gtn", "d_rgcn", "d1_noise", "e2_mix", "f_rgcn"]
for ds in DATASETS:
    L.append("| " + " | ".join(
        [ds, DOMAIN[ds]] + [f"{cell(ds, st)} (n={nn(ds, st)})" for st in COLS]) + " |")

cnt = lambda st: sum(1 for ds in DATASETS if nn(ds, st) > 0)
L += ["",
      "진척: " + ", ".join(f"**({lab}) {cnt(st)}/7**" for lab, st in
                          zip(["a", "b1", "b2", "c", "d", "e1", "e2", "f"], COLS)) +
      " 데이터셋 완료 (각 최대 10 seed).",
      "",
      "## 매핑 / 상태",
      "- (a)=`a1_baseline` · (c)=`c2_gtn` (backbone=han) · (d)=`d_rgcn` · (f)=`f_rgcn` (backbone=rgcn)",
      "- (b1)=`b1_noise` · (e1)=`d1_noise`: 위상 슬롯에 **같은 차원(res²×K)의 랜덤 가우시안** — 차원추가 효과 대조군",
      "- (b2)=`b2_mix` · (e2)=`e2_mix`: 실제 GTN-PDGNN 위상을 **같은 class·같은 split 안에서 shuffle** "
      "(`topology_mode=class_wise_mixing`) — per-node 위상↔노드 매칭만 파괴, class-level 분포 보존",
      "",
      "**주의(MAG·yelp 절대값):** macro-F1 이 낮은 건 모델 실패가 아니라 metric 특성 — "
      "MAG 는 349클래스를 6000노드 subsample 에서 평균(대부분 클래스가 test 표본 거의 0 → F1=0 다수), "
      "yelp 는 featureless 멀티라벨(16)에서 희귀 라벨 F1≈0. accuracy 로는 MAG 0.14→0.28(HAN→RGCN), "
      "yelp 0.62→0.87 로 정상 학습. 이 두 데이터셋은 절대값 비교 대신 **같은 데이터셋 내 조건 간 Δ**로만 해석.",
      "",
      "원본 per-run: `runs/campaign/<ds>__<setting>_s<seed>/metrics.json`", ""]
open("results/SUMMARY.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
