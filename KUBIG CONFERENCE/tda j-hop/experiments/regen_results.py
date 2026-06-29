"""runs/ 의 최신 metrics 를 results/ 로 모으고 results/SUMMARY.md 재생성.
멀티도메인(있는 데이터셋 전부) + ACM ablation 표를 만든다. 새 데이터셋이 늘면 자동 반영."""
import glob
import json
import os
import shutil

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
os.makedirs("results/multidomain", exist_ok=True)
os.makedirs("results/acm_ablation", exist_ok=True)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# --- copy latest metrics into results/ ---
for d in glob.glob("runs/multi/*/"):
    n = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/metrics.json"):
        shutil.copy(f"{d}/metrics.json", f"results/multidomain/{n}.json")
for d in glob.glob("runs/abl/*/"):
    n = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/metrics.json"):
        shutil.copy(f"{d}/metrics.json", f"results/acm_ablation/{n}.json")

DOMAIN = {"acm": "학술/인용", "dblp": "학술/인용", "imdb": "영화(멀티라벨)",
          "freebase": "지식그래프", "aminer": "학술(대형·subsample)", "mag": "학술(대형·subsample)",
          "ogbmag": "학술(대형·subsample)", "rcdd": "금융/이커머스"}

md = {os.path.basename(p)[:-5]: load(p) for p in glob.glob("results/multidomain/*.json")}
datasets = sorted(set(k.rsplit("_", 1)[0] for k in md))
lines = ["# TDA 실험 결과\n",
         "노드 분류, test Macro-F1 (성능 주장이 아니라 실측치). 원본 metrics 는 `multidomain/`, `acm_ablation/`.\n",
         "## 1. 여러 도메인 (baseline=HAN 단독 vs full=GTN+PDGNN+HAN)\n",
         "| 데이터셋 | 도메인 | baseline | full | Δ |",
         "|----------|--------|----------|------|---|"]
for ds in datasets:
    b = md.get(f"{ds}_baseline"); f = md.get(f"{ds}_full")
    bs = f"{b['test_macro_f1']:.4f}" if b else "—"
    fs = f"{f['test_macro_f1']:.4f}" if f else "—"
    dd = f"{f['test_macro_f1']-b['test_macro_f1']:+.4f}" if (b and f) else "—"
    lines.append(f"| {ds.upper()} | {DOMAIN.get(ds,'?')} | {bs} | {fs} | {dd} |")

# --- ACM ablation ---
def col(prefix, key="test_macro_f1"):
    xs = [load(p)[key] for p in glob.glob(f"results/acm_ablation/{prefix}_s*.json")
          if load(p) and key in load(p)]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (float("nan"), float("nan"), 0)

lines += ["", "## 2. ACM Ablation (seed 0/1/2)\n", "| 실험 | 결과 |", "|------|------|"]
for pre, lab in [("a1_baseline", "A1 HAN 단독"), ("c2_gtn", "C2 GTN+PDGNN+HAN"),
                 ("b2_manual", "B2 manual+EPD"), ("d2_nomin", "D2 no-MIN"),
                 ("d3_ch2", "D3 채널2"), ("d3_ch8", "D3 채널8"), ("d4_l1", "D4 깊이1"),
                 ("d4_l3", "D4 깊이3"), ("d5_random", "D5 random"), ("diag_topoonly", "topology-only")]:
    m, s, n = col(pre)
    if n:
        lines.append(f"| {lab} | {m:.4f} ± {s:.4f} |")
a3 = col("c2_gtn", "gtn_only_test_macro_f1")
if a3[2]:
    lines.append(f"| A3 GTN 단독 | {a3[0]:.4f} ± {a3[1]:.4f} |")
pm = col("c2_gtn", "test_macro_f1_permuted")
if pm[2]:
    lines.append(f"| permutation(위상셔플) | {pm[0]:.4f} |")
lines.append("| D1 no-kNN | 측정불가(>2h) → kNN 필수 |")

open("results/SUMMARY.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
