"""main.py - 강원도 전봇대 산불 위험 예측 파이프라인 전체 실행"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import OUT_DIR, TERRAIN_CSV


def main():
    print("=" * 60)
    print("강원도 전봇대 산불 위험 예측 파이프라인")
    print("=" * 60)

    # 1. AWS 전처리
    print("\n[1/7] AWS 전처리...")
    from preprocessing.aws_prep import run as aws_run, load_aws_meta
    aws_df, aws_meta = aws_run()

    # 2. ASOS 전처리
    print("\n[2/7] ASOS 전처리...")
    from preprocessing.asos_prep import run as asos_run, load_asos_meta
    asos_df, asos_meta = asos_run()

    # 3. FWI 계산 (전 기간 연속)
    print("\n[3/7] FWI 계산...")
    from preprocessing.fwi_prep import run as fwi_run
    aws_df = fwi_run(aws_df)

    # 4. AWS-ASOS 병합 + 시즌 필터
    print("\n[4/7] 병합 + 시즌 필터...")
    from preprocessing.merge import run as merge_run
    merged, season_df, poles = merge_run(aws_df, aws_meta, asos_df, asos_meta)

    # 5. Isolation Forest
    print("\n[5/7] Isolation Forest...")
    from model.isolation_forest import run as if_run
    station_scores = if_run(season_df)

    # anomaly score → poles 배정
    poles = poles.merge(
        station_scores.rename(columns={"지점": "nearest_aws_id"}),
        on="nearest_aws_id", how="left"
    )
    med_score = poles["max_anomaly_score"].median()
    poles["max_anomaly_score"] = poles["max_anomaly_score"].fillna(med_score)

    # terrain_weight 병합
    terrain = pd.read_csv(TERRAIN_CSV)[["pole_id","terrain_weight"]]
    poles = poles.merge(terrain, on="pole_id", how="left")
    poles["terrain_weight"] = poles["terrain_weight"].fillna(0.684)
    poles["adjusted_score"] = poles["max_anomaly_score"] * poles["terrain_weight"]
    poles.to_parquet(OUT_DIR / "poles_aws_score.parquet", index=False)

    # 6. 낙뢰 집계
    print("\n[6/7] 낙뢰 집계...")
    from preprocessing.lightning_prep import run as lightning_run
    poles = lightning_run(poles)

    # 7. 격자 클러스터 + 판정
    print("\n[7/7] 격자 클러스터 + 판정...")
    from model.grid_cluster import run as cluster_run
    poles_final = cluster_run(poles, T_percentile=80, min_cluster_size=10, K=20)

    # 검증
    print("\n[검증] 과거 산불 precision/recall...")
    from validation.wildfire_eval import run as eval_run
    p, r, f2 = eval_run(poles_final, radius_m=3000)

    print("\n" + "=" * 60)
    print(f"파이프라인 완료")
    print(f"  decision=1: {poles_final['decision'].sum():,}개 전봇대")
    print(f"  Precision={p:.4f}  Recall={r:.4f}  F2={f2:.4f}")
    print(f"  결과 저장: {OUT_DIR}")
    print("=" * 60)
    return poles_final


if __name__ == "__main__":
    main()
