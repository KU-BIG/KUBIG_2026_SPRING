"""14개 데이터셋을 실제 로드해서 그래프 통계(노드타입·엣지타입·타겟·클래스·feature)를 측정.
큰 데이터셋 로더(big.py)는 내부에서 subsample하므로, 측정값은 '로드된 그래프' 기준이며
원본 cap은 big.py / config 의 값으로 별도 표기한다. CPU 전용(GPU 미사용)."""
import json
import os
import sys

sys.path.insert(0, "/mnt/data/users/junyoungpark/code/TDA")
from tda.data.registry import DATASETS  # noqa: E402

DATA_ROOT = "/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data"
ORDER = ["acm", "dblp", "imdb", "freebase", "aminer", "mag", "dblp_pyg", "imdb_pyg",
         "aifb", "mutag", "bgs", "am", "pubmed", "yelp"]

out = {}
for name in ORDER:
    try:
        data, target = DATASETS[name](DATA_ROOT)
        ntypes = {nt: int(data[nt].num_nodes) for nt in data.node_types}
        etypes = {"__".join(et): int(data[et].num_edges) for et in data.edge_types}
        tgt = data[target]
        y = tgt.y
        if y.dim() > 1:
            nc, ml = int(y.shape[1]), True
        else:
            nc, ml = int(y[y >= 0].max()) + 1, False
        feat = "featureless" if getattr(tgt, "x", None) is None else int(tgt.x.shape[1])
        rec = {"target": target, "n_target": int(tgt.num_nodes),
               "node_types": ntypes, "total_nodes": sum(ntypes.values()),
               "edge_types": etypes, "total_edges": sum(etypes.values()),
               "classes": nc, "feat": feat, "multilabel": ml}
        out[name] = rec
        print(f"OK {name}: tgt={target}({tgt.num_nodes}) Ntypes={len(ntypes)} "
              f"N={sum(ntypes.values())} Etypes={len(etypes)} E={sum(etypes.values())} "
              f"cls={nc} feat={feat} ml={ml}", flush=True)
    except Exception as e:
        out[name] = {"error": str(e)}
        print(f"FAIL {name}: {e}", flush=True)

json.dump(out, open("/mnt/data/users/junyoungpark/code/TDA/experiments/dataset_stats.json", "w"),
          indent=2)
print("WROTE experiments/dataset_stats.json", flush=True)
