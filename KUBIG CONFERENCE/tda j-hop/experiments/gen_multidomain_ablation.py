"""8개 데이터셋 × ablation(B/D 계열) config 생성 → configs/abl_multi/*.json + manifest.
A1/C2 는 multidomain 브레드스에서 이미 돌았으므로 여기선 B2·D2·D3·D4·D5 만 생성한다."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/abl_multi"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["acm", "dblp", "imdb", "freebase", "aminer", "mag", "dblp_pyg", "imdb_pyg"]
# (suffix, [(dotted-key, value)]) — B/D 계열 (A1/C2 는 이미 있음, D1 은 비현실적이라 제외)
SETTINGS = [
    ("b2_manual", [("topology_source", "manual")]),
    ("d2_nomin", [("topology_source", "gtn"), ("pdgnn.agg", "sum")]),
    ("d3_ch8", [("topology_source", "gtn"), ("gtn.num_channels", 8)]),
    ("d4_l1", [("topology_source", "gtn"), ("gtn.num_layers", 1)]),
    ("d5_random", [("topology_source", "random")]),
]


def deep(d, path, val):
    ks = path.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = val


names = []
for ds in DATASETS:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, ov in SETTINGS:
        cfg = copy.deepcopy(base)
        for k, v in ov:
            deep(cfg, k, v)
        path = f"{OUT}/{ds}__{suf}.json"
        json.dump(cfg, open(path, "w"), indent=2)
        names.append(path)
open(f"{OUT}/manifest.txt", "w").write("\n".join(names) + "\n")
print(f"wrote {len(names)} configs ({len(DATASETS)} datasets x {len(SETTINGS)} settings)")
