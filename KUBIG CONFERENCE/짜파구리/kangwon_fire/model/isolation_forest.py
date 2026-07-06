"""Isolation Forest 학습 + 지점별 시즌 max anomaly score"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES, OUT_DIR


def run(season_df):
    available = [f for f in FEATURES if f in season_df.columns]
    missing   = [f for f in FEATURES if f not in season_df.columns]
    if missing:
        print(f"  [WARN] 누락 피처: {missing}")
    print(f"[IF] 학습 행렬: {len(season_df):,}행 × {len(available)}피처")

    X = season_df[available].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    print("[IF] IsolationForest 학습 (n=300, contamination=0.05)...")
    model = IsolationForest(n_estimators=300, contamination=0.05,
                            max_samples="auto", random_state=42, n_jobs=-1)
    model.fit(Xs)

    season_df = season_df.copy()
    season_df["anomaly_score"] = -model.score_samples(Xs)

    station_score = (season_df.groupby("지점")["anomaly_score"]
                     .max().reset_index()
                     .rename(columns={"anomaly_score": "max_anomaly_score"}))

    joblib.dump(model,  OUT_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    station_score.to_parquet(OUT_DIR / "station_scores.parquet", index=False)
    season_df.to_parquet(OUT_DIR / "season_scored.parquet", index=False)
    print(f"[IF] max score: min={station_score['max_anomaly_score'].min():.4f} "
          f"median={station_score['max_anomaly_score'].median():.4f} "
          f"max={station_score['max_anomaly_score'].max():.4f}")
    return station_score


if __name__ == "__main__":
    season_df = pd.read_parquet(OUT_DIR / "merged_season.parquet")
    run(season_df)
