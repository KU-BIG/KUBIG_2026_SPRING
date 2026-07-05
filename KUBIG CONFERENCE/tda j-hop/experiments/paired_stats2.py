"""주입 factorial(gated: RGCN/HAN) + LP(L1/L2) paired per-seed 검정 → results/paired_stats2.md"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
DS = ["acm", "dblp", "aifb"]


def load(pattern, key):
    out = {}
    for p in glob.glob(pattern):
        s = int(p.split("_s")[-1].split("/")[0])
        v = json.load(open(p)).get(key)
        if v is not None and np.isfinite(v):
            out[s] = v
    return out


def paired(a, b):
    """Δ = b − a (seed 짝)."""
    ss = sorted(set(a) & set(b))
    if len(ss) < 5:
        return None
    d = np.array([b[s] - a[s] for s in ss])
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    try:
        p = st.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    return dict(n=len(d), md=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci,
                win=float((d > 0).mean()), p=p)


def row(title, a, b):
    r = paired(a, b)
    if r is None:
        return f"| {title} | – | (짝 부족) | – | – |"
    sig = " **\\***" if r["p"] < 0.05 else ""
    return (f"| {title} | {r['n']} | {r['md']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
            f"{r['win']:.0%} | {r['p']:.3g}{sig} |")


L = ["# Paired 검정 v2 — 주입 factorial(gated) + LP\n"]
HDR = "| 대조 | n | mean Δ [95% CI] | win | Wilcoxon p |\n|---|---|---|---|---|"

for bk, pre in [("RGCN", "gt"), ("HAN", "gh")]:
    L.append(f"\n## 주입 factorial — {bk} (test macro-F1)\n")
    for ds in DS:
        g = lambda c: load(f"runs/gated/{ds}__{pre}_{c}_s*/metrics.json", "test_macro_f1")
        base, cr, cn_, cm = g("base"), g("cat_real"), g("cat_noise"), g("cat_mix")
        gr, gn, gm = g("gate_real"), g("gate_noise"), g("gate_mix")
        L.append(f"### {ds}\n" + HDR)
        L.append(row("cat_real − base", base, cr))
        L.append(row("gate_real − base", base, gr))
        L.append(row("gate_real − cat_real (주입효과)", cr, gr))
        L.append(row("gate_noise − base", base, gn))
        L.append(row("gate_real − gate_mix (정렬)", gm, gr))
        L.append("")

L.append("\n## LP Level 1 (test AUC, 5 seed — 저검정력 주의)\n")
for ds in DS:
    g = lambda c: load(f"runs/lp/{ds}__lp_{c}_s*/metrics.json", "test_auc")
    a, b1, c, m = g("a"), g("b1"), g("c"), g("m")
    L.append(f"### {ds}\n" + HDR)
    L.append(row("node-PI(c) − base", a, c))
    L.append(row("node-PI(c) − noise(b1)", b1, c))
    L.append("")

L.append("\n## LP Level 2 (test AUC, 10 seed)\n")
for ds in DS:
    g = lambda c: load(f"runs/lp/{ds}__lp2_{c}_s*/metrics.json", "test_auc")
    base, real, noise, mix = g("base"), g("real"), g("noise"), g("mix")
    L.append(f"### {ds}\n" + HDR)
    L.append(row("pair-PI(real) − base", base, real))
    L.append(row("real − noise", noise, real))
    L.append(row("real − mix(CN)", mix, real))
    L.append("")

# aifb HAN base 분포 진단 (baseline 재평가 근거)
gh = load("runs/gated/aifb__gh_base_s*/metrics.json", "test_macro_f1")
v = np.array(sorted(gh.values()))
L.append(f"\n## 진단: aifb gh_base(커스텀 attention HAN) per-seed 분포\n\n"
         f"n={len(v)}, min={v.min():.3f}, median={np.median(v):.3f}, max={v.max():.3f}, "
         f"values={[round(x,3) for x in v]}\n")
open("results/paired_stats2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
