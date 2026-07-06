import sys
sys.path.insert(0, 'kangwon_fire')
import pandas as pd
from config import OUT_DIR
from validation.wildfire_eval import load_wildfire, eval_metrics
from model.grid_cluster import run as cluster_run

poles_base = pd.read_parquet(OUT_DIR / 'poles_aws_score.parquet')
# lightning 컬럼이 없으면 poles_lightning.parquet 사용
try:
    poles_base = pd.read_parquet(OUT_DIR / 'poles_lightning.parquet')
    print(f'poles_lightning 로드: {len(poles_base):,}행')
except FileNotFoundError:
    print(f'poles_aws_score 로드: {len(poles_base):,}행')

fires = load_wildfire()

T_range   = [70, 75, 80, 85, 90]
n_range   = [5, 10, 20]
K_range   = [10, 15, 20, 25, 30]
RADIUS_M  = 3000

best = {"f2": -1, "params": {}}
results = []
total = len(T_range) * len(n_range) * len(K_range)
print(f"\n[GridSearch] 총 {total}개 조합, radius={RADIUS_M}m\n")

for i, T in enumerate(T_range):
    for n in n_range:
        for K in K_range:
            poles = cluster_run(poles_base.copy(), T_percentile=T,
                                min_cluster_size=n, K=K)
            p, r, f2 = eval_metrics(poles, fires, radius_m=RADIUS_M)
            results.append({"T": T, "n": n, "K": K,
                            "precision": p, "recall": r, "f2": f2,
                            "decision1": int(poles["decision"].sum())})
            if f2 > best["f2"]:
                best = {"f2": f2, "params": {"T": T, "n": n, "K": K}}
            idx = len(results)
            if idx % 10 == 0 or idx == total:
                print(f"  [{idx}/{total}] best F2={best['f2']:.4f} @ {best['params']}")

res_df = pd.DataFrame(results).sort_values("f2", ascending=False)
res_df.to_csv(OUT_DIR / "grid_search_results.csv", index=False)
print(f"\n[GridSearch] 최적: F2={best['f2']:.4f} @ {best['params']}")
print("\n상위 10개 조합:")
print(res_df.head(10).to_string(index=False))
