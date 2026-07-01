"""Class-wise topology mixing ablation configs — 7 datasets x {HAN,RGCN}.

This keeps the existing GTN-PDGNN topology pipeline, then shuffles the computed topology
features within the same class and split before concatenating them to node features.
"""
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
    ("han__class_wise_mixing", {"backbone": "han"}),
    ("rgcn__class_wise_mixing", {"backbone": "rgcn", "rgcn": RGCN}),
]

names = []
for ds in DATASETS:
    base = json.load(open(f"configs/{ds}.json"))
    for suf, ov in SETTINGS:
        cfg = copy.deepcopy(base)
        cfg.update({
            "use_topology": True,
            "topology_source": "gtn",
            "topology_mode": "class_wise_mixing",
            "class_wise_mixing_by_split": True,
            "class_wise_mixing_multilabel_fallback": "auto",
        })
        if "rgcn" in ov:
            cfg["rgcn"] = copy.deepcopy(ov["rgcn"])
        cfg["backbone"] = ov["backbone"]
        path = f"{OUT}/{ds}__{suf}.json"
        json.dump(cfg, open(path, "w"), indent=2)
        names.append(path)

manifest = f"{OUT}/class_wise_mixing_manifest.txt"
open(manifest, "w").write("\n".join(names) + "\n")
print(f"wrote {len(names)} class-wise mixing configs ({len(DATASETS)} datasets x 2 backbones)")
print(f"manifest: {manifest}")
print("\n".join("  " + n for n in names))
