"""실험 ① config 생성 — {acm, dblp, aifb} × 7조건 (base + 주입2 × 내용3)."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/campaign"
os.makedirs(OUT, exist_ok=True)

GATED = {"hidden": 64, "num_layers": 2, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
         "epochs": 100, "gate_hidden": 64, "topo_seed": 777, "topo_cache": "cache/topo_manual"}
COND = [("gt_base", "none", "real"),
        ("gt_cat_real", "concat", "real"), ("gt_cat_noise", "concat", "noise"),
        ("gt_cat_mix", "concat", "mix"),
        ("gt_gate_real", "gate", "real"), ("gt_gate_noise", "gate", "noise"),
        ("gt_gate_mix", "gate", "mix")]

names = []
for ds in ["acm", "dblp", "aifb", "imdb", "freebase", "mag", "yelp"]:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, inj, cont in COND:
        cfg = copy.deepcopy(base)
        cfg["gated"] = dict(GATED, injection=inj, content=cont)
        json.dump(cfg, open(f"{OUT}/{ds}__{suf}.json", "w"), indent=2)
        names.append(f"{OUT}/{ds}__{suf}.json")
        # HAN 백본 판 (gh_*) — 동일 factorial, backbone 만 교체
        cfg2 = copy.deepcopy(cfg)
        cfg2["gated"]["backbone"] = "han"
        suf2 = suf.replace("gt_", "gh_")
        json.dump(cfg2, open(f"{OUT}/{ds}__{suf2}.json", "w"), indent=2)
        names.append(f"{OUT}/{ds}__{suf2}.json")
print(f"wrote {len(names)} gated configs (rgcn gt_* + han gh_*)")

# ---- 최종 factorial (gated2) — 위상 = 고정 GTN+PDGNN (channels=gtn_fixed) ----
# base/noise 는 위상 무관이라 gt_/gh_ 재사용, real/mix 만 gt2_/gh2_ (freebase·yelp 제외 5 ds).
COND2 = [("gt2_cat_real", "concat", "real"), ("gt2_cat_mix", "concat", "mix"),
         ("gt2_gate_real", "gate", "real"), ("gt2_gate_mix", "gate", "mix")]
n2 = 0
for ds in ["acm", "dblp", "imdb", "mag", "aifb"]:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, inj, cont in COND2:
        cfg = copy.deepcopy(base)
        cfg["gated"] = dict(GATED, topo_cache="cache/topo_gtnfix", channels="gtn_fixed",
                            injection=inj, content=cont)
        json.dump(cfg, open(f"{OUT}/{ds}__{suf}.json", "w"), indent=2)
        cfg2 = copy.deepcopy(cfg)
        cfg2["gated"]["backbone"] = "han"
        json.dump(cfg2, open(f"{OUT}/{ds}__{suf.replace('gt2_', 'gh2_')}.json", "w"), indent=2)
        n2 += 2
print(f"wrote {n2} gated2 configs (gtn_fixed: gt2_* + gh2_*)")
