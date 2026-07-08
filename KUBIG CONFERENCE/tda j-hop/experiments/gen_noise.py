"""noisy-topology 대조군 config 생성 (7개 데이터셋).
  B1 (b1_noise): HAN  + 위상 슬롯 = random noise   (topology_noise=random)
  D1 (d1_noise): RGCN + 위상 슬롯 = random noise
B2/D2 (class-wise mixing)는 실제 위상이 필요해 별도 생성(gen_noise_classmix)."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/campaign"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
RGCN = {"hidden_dim": 64, "num_layers": 2, "num_bases": None,
        "lr": 0.01, "weight_decay": 0.0005, "dropout": 0.5, "epochs": 100}
SETTINGS = [
    ("b1_noise", {"backbone": "han", "use_topology": True, "topology_noise": "random"}, False),
    ("d1_noise", {"backbone": "rgcn", "use_topology": True, "topology_noise": "random"}, True),
]

n = 0
for ds in DATASETS:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, ov, is_rgcn in SETTINGS:
        cfg = copy.deepcopy(base)
        if is_rgcn:
            cfg["rgcn"] = dict(RGCN)
        cfg.update(ov)
        json.dump(cfg, open(f"{OUT}/{ds}__{suf}.json", "w"), indent=2)
        n += 1
print(f"wrote {n} configs (b1_noise, d1_noise) x {len(DATASETS)} datasets")
