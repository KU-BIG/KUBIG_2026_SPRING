"""통합 캠페인: 8 데이터셋 × 전체 A~D 설정 → configs/campaign/*.json + manifest.
SLURM 배열이 (config × seed) 로 240 task 를 큐에 쌓아 8 GPU 를 계속 포화시킨다."""
import copy
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(os.path.abspath(ROOT))
OUT = "configs/campaign"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["acm", "dblp", "imdb", "freebase", "aminer", "mag", "dblp_pyg", "imdb_pyg",
            "aifb", "mutag", "bgs", "am", "pubmed", "yelp"]  # 14개 (학술/영화/지식/RDF/생의학/business)
# (suffix, overrides). A1/B/C/D 전부 (A3 는 c2 안에 기록, D1=no-kNN 은 비현실적이라 제외).
SETTINGS = [
    ("a1_baseline", [("use_topology", False)]),
    ("c2_gtn", [("topology_source", "gtn"), ("permute_topology", True)]),
    ("b2_manual", [("topology_source", "manual")]),
    ("d2_nomin", [("topology_source", "gtn"), ("pdgnn.agg", "sum")]),
    ("d3_ch2", [("topology_source", "gtn"), ("gtn.num_channels", 2)]),
    ("d3_ch8", [("topology_source", "gtn"), ("gtn.num_channels", 8)]),
    ("d4_l1", [("topology_source", "gtn"), ("gtn.num_layers", 1)]),
    ("d4_l3", [("topology_source", "gtn"), ("gtn.num_layers", 3)]),
    ("d5_random", [("topology_source", "random")]),
    ("topoonly", [("topology_source", "gtn"), ("node_features", "off")]),
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
print(f"wrote {len(names)} configs ({len(DATASETS)} ds x {len(SETTINGS)} settings); "
      f"x3 seeds = {len(names)*3} tasks")
