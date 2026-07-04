"""핵심 대조들의 paired(seed별 짝) 통계: mean Δ, 95% CI, win-rate, Wilcoxon p.
데이터 출처: 로컬 runs(a,b1,c,d,e1) + CLASS_WISE_MIXING.md(b2,e2) + f_rgcn_per_seed.md(f 4개 ds).
출력: results/paired_stats.md"""
import glob
import json
import re

import numpy as np
from scipy import stats as st

DS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
SEEDS = [int(x) for x in open("experiments/seeds.txt").read().split()]

# ---- 1) 로컬 runs ----
def load_local(setting):
    out = {}
    for ds in DS:
        d = {}
        for p in glob.glob(f"runs/campaign/{ds}__{setting}_s*/metrics.json"):
            s = int(p.split("_s")[-1].split("/")[0])
            if s in SEEDS:
                v = json.load(open(p)).get("test_macro_f1")
                if v is not None and np.isfinite(v):
                    d[s] = v
        out[ds] = d
    return out

M = {k: load_local(k) for k in ["a1_baseline", "b1_noise", "c2_gtn", "d_rgcn", "d1_noise"]}

# ---- 2) class-mix per-seed (CLASS_WISE_MIXING.md 상세표) ----
M["b2"], M["e2"] = {ds: {} for ds in DS}, {ds: {} for ds in DS}
for ln in open("results/CLASS_WISE_MIXING.md"):
    m = re.match(r"\|\s*(\w+)\s*\|\s*(HAN|RGCN)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|", ln)
    if m:
        ds, bb, s, f1 = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        if ds in DS and s in SEEDS:
            M["b2" if bb == "HAN" else "e2"][ds][s] = f1

# ---- 3) f per-seed (f_rgcn_per_seed.md) ----
M["f"] = {ds: {} for ds in DS}
txt = open("results/f_rgcn_per_seed.md").read()
for sec in re.split(r"\n## ", txt)[1:]:
    name = sec.split("\n")[0].strip().lower()
    if name not in DS:
        continue
    for m in re.finditer(r"^\|\s*(\d{4,6})\s*\|\s*([0-9.]+)\s*\|", sec, re.M):
        s = int(m.group(1))
        if s in SEEDS:
            M["f"][name][s] = float(m.group(2))

# ---- paired 대조 ----
def paired(ds, x, y):
    """Δ = M[y] − M[x] per seed."""
    a, b = M[x].get(ds, {}), M[y].get(ds, {})
    ss = sorted(set(a) & set(b))
    if len(ss) < 6:
        return None
    d = np.array([b[s] - a[s] for s in ss])
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    try:
        p = st.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    return dict(n=len(d), md=d.mean(), lo=d.mean()-ci, hi=d.mean()+ci,
                win=float((d > 0).mean()), p=p)

CONTRASTS = [
    ("c2_gtn",  "a1_baseline", "(c)−(a): HAN 위상 이득"),
    ("b1_noise", "a1_baseline", "(b1)−(a): noise 효과"),
    ("c2_gtn",  "b1_noise", "(c)−(b1): 위상 vs noise", True),   # y-x 반전용 표기만
    ("b2",      "c2_gtn", "(b2)−(c): mix − real (HAN)"),
    ("f",       "d_rgcn", "(f)−(d): RGCN 위상 이득"),
    ("d1_noise", "d_rgcn", "(e1)−(d): RGCN noise 효과"),
    ("e2",      "f", "(e2)−(f): mix − real (RGCN)"),
    ("d_rgcn",  "a1_baseline", "(d)−(a): RGCN − HAN"),
]

L = ["# Paired per-seed 통계 (test Macro-F1)\n",
     "Δ = 뒤조건 − 앞조건 이 아니라 **표기된 방향** (mean Δ [95% CI], win-rate=Δ>0 비율, Wilcoxon p, n=짝 seed 수).",
     "b2/e2 는 CLASS_WISE_MIXING.md, f 는 f_rgcn_per_seed.md, 나머지는 로컬 runs 에서 seed 짝 매칭.\n"]
for spec in CONTRASTS:
    x, y, title = spec[0], spec[1], spec[2]
    L.append(f"## {title}\n")
    L.append("| 데이터셋 | n | mean Δ [95% CI] | win-rate | Wilcoxon p |")
    L.append("|---|---|---|---|---|")
    for ds in DS:
        r = paired(ds, y, x)  # Δ = M[x] − M[y] : x가 '앞'(왼쪽 항)
        if r is None:
            L.append(f"| {ds} | – | (짝 데이터 없음) | – | – |")
            continue
        sig = " **\\***" if r["p"] < 0.05 else ""
        L.append(f"| {ds} | {r['n']} | {r['md']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                 f"{r['win']:.0%} | {r['p']:.3g}{sig} |")
    L.append("")
open("results/paired_stats.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- 최악-seed(robustness) ----
print("\n===== worst-seed (min over seeds) macro-F1 =====")
for ds in DS:
    row = [ds]
    for k in ["a1_baseline", "c2_gtn", "d_rgcn", "f"]:
        v = M[k].get(ds, {})
        row.append(f"{min(v.values()):.3f}" if v else "–")
    print("  " + " | ".join(row))
