"""새 설계 (d)/(f) config 생성 — RGCN backbone. 7개 데이터셋.
  (d) d_rgcn : backbone=rgcn, use_topology=False         (RGCN only)
  (f) f_rgcn : backbone=rgcn, topology_source=gtn         (RGCN + GTN-PDGNN feature)
(a)=a1_baseline, (c)=c2_gtn 는 기존 config/결과 재사용."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/campaign"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
# RGCN 하이퍼파라미터(가정): HAN 과 비슷한 예산. num_bases=null → 관계별 full weight.
RGCN = {"hidden_dim": 64, "num_layers": 2, "num_bases": None,
        "lr": 0.01, "weight_decay": 0.0005, "dropout": 0.5, "epochs": 100}
SETTINGS = [
    ("d_rgcn", {"backbone": "rgcn", "use_topology": False}),
    ("f_rgcn", {"backbone": "rgcn", "use_topology": True, "topology_source": "gtn"}),
]

names = []
for ds in DATASETS:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, ov in SETTINGS:
        cfg = copy.deepcopy(base)
        cfg["rgcn"] = dict(RGCN)
        cfg.update(ov)
        path = f"{OUT}/{ds}__{suf}.json"
        json.dump(cfg, open(path, "w"), indent=2)
        names.append(path)
print(f"wrote {len(names)} configs (d_rgcn, f_rgcn) x {len(DATASETS)} datasets")
print("\n".join("  " + n for n in names))
