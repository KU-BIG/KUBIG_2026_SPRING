import sys
sys.path.insert(0, 'kangwon_fire')
from validation.wildfire_eval import run, load_wildfire
import pandas as pd
from config import OUT_DIR

poles = pd.read_parquet(OUT_DIR / 'poles_decision.parquet')
print(f'poles 로드: {len(poles):,}행, decision=1: {poles["decision"].sum():,}개')

fires = load_wildfire()

print("\n=== 반경별 성능 비교 ===")
for r in [500, 1000, 2000, 3000, 5000]:
    from validation.wildfire_eval import eval_metrics
    p, rec, f2 = eval_metrics(poles, fires, radius_m=r)
    ok = "✓" if f2 >= 0.30 else "✗"
    print(f"  radius={r:5d}m  P={p:.4f}  R={rec:.4f}  F2={f2:.4f} {ok}")
