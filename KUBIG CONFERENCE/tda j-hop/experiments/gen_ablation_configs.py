"""ACM ablation config 생성기. configs/acm.json 을 베이스로 override 적용 -> configs/ablation/*.json.

플랜 Group D + 진단을 매핑한다. 각 설정은 SLURM 배열에서 seed 0/1/2 로 실행된다.
A3(GTN 단독)은 c2_gtn 실행 안에서 gtn_only_test_macro_f1 로 기록된다(별도 실행 불필요).
"""
import copy
import json
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "configs", "acm.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "configs", "ablation")


def deep(d, path, val):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = val


# (name, [(dotted-key, value), ...])
SETTINGS = [
    ("a1_baseline", [("use_topology", False)]),                                  # A1
    ("b2_manual", [("topology_source", "manual")]),                              # B2 (= D6 수동)
    ("c2_gtn", [("topology_source", "gtn"), ("permute_topology", True)]),        # C2 (+A3, +permutation 진단)
    ("d1_noknn", [("topology_source", "gtn"), ("pdgnn.channel_knn_k", 100000)]), # D1
    ("d2_nomin", [("topology_source", "gtn"), ("pdgnn.agg", "sum")]),            # D2
    ("d3_ch2", [("topology_source", "gtn"), ("gtn.num_channels", 2)]),           # D3 (C=4 는 c2_gtn)
    ("d3_ch8", [("topology_source", "gtn"), ("gtn.num_channels", 8)]),           # D3
    ("d4_l1", [("topology_source", "gtn"), ("gtn.num_layers", 1)]),              # D4 (L=2 는 c2_gtn)
    ("d4_l3", [("topology_source", "gtn"), ("gtn.num_layers", 3)]),              # D4
    ("d5_random", [("topology_source", "random")]),                             # D5
    ("diag_topoonly", [("topology_source", "gtn"), ("node_features", "off")]),   # 진단: topology-only
]


def main():
    base = json.load(open(os.path.abspath(BASE)))
    os.makedirs(os.path.abspath(OUT), exist_ok=True)
    names = []
    for name, overrides in SETTINGS:
        cfg = copy.deepcopy(base)
        for k, v in overrides:
            deep(cfg, k, v)
        path = os.path.join(os.path.abspath(OUT), f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        names.append(f"configs/ablation/{name}.json")
    with open(os.path.join(os.path.abspath(OUT), "manifest.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} ablation configs to {OUT}")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
