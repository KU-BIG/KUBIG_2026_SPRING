"""LP Level-1 config 생성 — 3 소형 데이터셋 × 조건 {a, b1, c, m} (+dblp c').
docs/lp_design.md. split/topo seed 고정 → 위상은 데이터셋당 1회 계산(캐시)."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/campaign"
os.makedirs(OUT, exist_ok=True)

# 주 타겟-타겟 관계 = 각 config 의 han_metapaths[0]
TARGET = {ds: json.load(open(f"configs/{ds}.json"))["han_metapaths"][0]
          for ds in ["acm", "dblp", "aifb", "imdb", "freebase", "mag", "yelp"]}
LP = {"hidden": 64, "num_layers": 2, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
      "epochs": 200, "edge_cap": 5000, "split_seed": 20260705, "topo_seed": 777,
      "topo_cache": "cache/topo_lp"}
COND = [
    ("lp_a", {"use_topology": False}),
    ("lp_b1", {"use_topology": True, "topology_noise": "random"}),
    ("lp_c", {"use_topology": True}),
    ("lp_m", {"use_topology": True, "topology_mode": "degree_mix"}),
]

names = []
for ds, tgt in TARGET.items():
    base = json.load(open(f"configs/{ds}.json"))
    for suf, ov in COND:
        cfg = copy.deepcopy(base)
        cfg["lp"] = dict(LP, target_relation=tgt)
        cfg.pop("topology_source", None)
        cfg.update(ov)
        path = f"{OUT}/{ds}__{suf}.json"
        json.dump(cfg, open(path, "w"), indent=2)
        names.append(path)
# c' — dblp: 위상 채널에서 예측 대상 관계(APA) 제외 → 신호 출처 분해
cfg = json.load(open(f"{OUT}/dblp__lp_c.json"))
cfg["lp"]["topo_exclude_target"] = True
json.dump(cfg, open(f"{OUT}/dblp__lp_cx.json", "w"), indent=2)
names.append(f"{OUT}/dblp__lp_cx.json")

# ---- Level 2 (pair-vicinity EPD, TLC-GNN 식 decoder 주입) ----
# encoder 는 위상 없음(use_topology=false) — pair 특징 효과만 격리. base 도 fixed-neg 프로토콜.
COND2 = [("lp2_base", "none"), ("lp2_real", "real"),
         ("lp2_noise", "noise"), ("lp2_mix", "mix")]
for ds, tgt in TARGET.items():
    base = json.load(open(f"configs/{ds}.json"))
    for suf, pk in COND2:
        cfg = copy.deepcopy(base)
        cfg["lp"] = dict(LP, target_relation=tgt, pair_feature=pk,
                         fixed_train_neg=True, pair_cache="cache/pair_epd")
        cfg["use_topology"] = False
        cfg.pop("topology_source", None)
        json.dump(cfg, open(f"{OUT}/{ds}__{suf}.json", "w"), indent=2)
        names.append(f"{OUT}/{ds}__{suf}.json")
print(f"wrote {len(names)} LP configs (L1 13 + L2 12)")
