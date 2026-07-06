"""merge.py - AWS-ASOS IDW 매칭 + 전봇대 AWS 배정 + 시즌 필터"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASOS_COLS, FIRE_SEASON_MONTHS, OUT_DIR, POLES_CSV


def match_aws_asos(aws_meta, asos_meta, asos_df, aws_df):
    print("[MERGE] AWS-ASOS IDW 매칭 (k=3, 30km 임계값)...")
    asos_coords = np.radians(asos_meta[["위도", "경도"]].values)
    tree = BallTree(asos_coords, metric="haversine")
    aws_coords = np.radians(aws_meta[["위도", "경도"]].values)
    dists, idxs = tree.query(aws_coords, k=min(3, len(asos_meta)))
    dists_km = dists * 6371

    extra_cols = ["effective_humidity", "dryness_index", "dry_wind_flag", "dry_wind_streak"]
    merge_cols = [c for c in ASOS_COLS if c not in ["지점", "일시", "일강수량(mm)", "일 최심신적설(cm)"]]
    merge_cols += [c for c in extra_cols if c in asos_df.columns]
    merge_cols = [c for c in merge_cols if c in asos_df.columns]

    parts = []
    for i, aws_row in enumerate(aws_meta.itertuples()):
        stn = aws_row.지점
        valid = dists_km[i] < 30
        if not valid.any():
            valid[0] = True
        vd = dists_km[i][valid]
        vi = idxs[i][valid]
        w = 1 / (vd ** 2 + 1e-9); w /= w.sum()

        matched_stns = asos_meta.iloc[vi]["지점"].values
        sub_aws = aws_df[aws_df["지점"] == stn][["일시"]].copy()
        combined = None
        for asos_stn, wt in zip(matched_stns, w):
            sub_a = asos_df[asos_df["지점"] == asos_stn][["일시"] + merge_cols].copy()
            for col in merge_cols:
                sub_a[col] = sub_a[col] * wt
            if combined is None:
                combined = sub_a
            else:
                combined = combined.set_index("일시").add(
                    sub_a.set_index("일시"), fill_value=0).reset_index()
        sub_aws = sub_aws.merge(combined, on="일시", how="left")
        sub_aws["지점"] = stn
        parts.append(sub_aws)

    asos_matched = pd.concat(parts, ignore_index=True)
    merged = aws_df.merge(asos_matched, on=["지점", "일시"], how="left")
    print(f"  병합: {len(merged):,}행")
    return merged


def assign_poles(poles, aws_meta):
    print("[MERGE] 전봇대→AWS 배정 (20km 플래그)...")
    aws_coords = np.radians(aws_meta[["위도", "경도"]].values)
    tree = BallTree(aws_coords, metric="haversine")
    pc = np.radians(poles[["lat", "lon"]].values)
    dists, idxs = tree.query(pc, k=1)
    dists_km = dists.flatten() * 6371
    poles = poles.copy()
    poles["nearest_aws_id"]  = aws_meta.iloc[idxs.flatten()]["지점"].values
    poles["aws_distance_km"] = dists_km.round(2)
    poles["aws_reliable"]    = dists_km < 20
    print(f"  20km 신뢰: {poles['aws_reliable'].sum():,} / {len(poles):,}")
    return poles


def run(aws_df, aws_meta, asos_df, asos_meta):
    merged = match_aws_asos(aws_meta, asos_meta, asos_df, aws_df)

    poles = pd.read_csv(POLES_CSV, encoding="cp949")[["pole_id","lon","lat"]].copy()
    poles = assign_poles(poles, aws_meta)

    season_df = merged[merged["일시"].dt.month.isin(FIRE_SEASON_MONTHS)].copy()
    print(f"[MERGE] 시즌 필터: {len(merged):,} → {len(season_df):,}행")

    merged.to_parquet(OUT_DIR / "merged_full.parquet", index=False)
    season_df.to_parquet(OUT_DIR / "merged_season.parquet", index=False)
    poles.to_parquet(OUT_DIR / "poles_base.parquet", index=False)
    print("[MERGE] 저장 완료")
    return merged, season_df, poles


if __name__ == "__main__":
    from preprocessing.aws_prep  import load_aws_meta
    from preprocessing.asos_prep import load_asos_meta
    aws_df   = pd.read_parquet(OUT_DIR / "aws_fwi.parquet")
    asos_df  = pd.read_parquet(OUT_DIR / "asos_processed.parquet")
    run(aws_df, load_aws_meta(), asos_df, load_asos_meta())
