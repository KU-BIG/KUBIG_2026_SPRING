"""낙뢰 BallTree 반경 2km 집계 (고려사항 §2: abs 최대값, 연간 전체)"""
import glob, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LIGHTNING_DIR, OUT_DIR, FIRE_SEASON_MONTHS


def load_lightning():
    files = sorted(glob.glob(str(LIGHTNING_DIR / "*.csv")))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="cp949"))
        except Exception as e:
            print(f"  [WARN] {Path(f).name}: {e}")
    raw = pd.concat(dfs, ignore_index=True)
    raw["일시"] = pd.to_datetime(raw["일시"], errors="coerce")
    raw = raw.dropna(subset=["일시", "위도", "경도"])
    raw["강도_abs"] = raw["낙뢰 강도"].abs()
    raw["month"] = raw["일시"].dt.month
    print(f"  낙뢰 raw: {len(raw):,}건")
    return raw


def run(poles):
    print("[LIGHTNING] 낙뢰 데이터 로드...")
    raw = load_lightning()

    pole_coords = np.radians(poles[["lat", "lon"]].values)
    tree = BallTree(pole_coords, metric="haversine")
    RADIUS_RAD = 2.0 / 6371

    lightning_coords = np.radians(raw[["위도", "경도"]].values)
    print(f"  BallTree query ({len(raw):,}건)...")
    indices_list = tree.query_radius(lightning_coords, r=RADIUS_RAD)

    count_all    = np.zeros(len(poles), dtype=np.int32)
    max_int_all  = np.zeros(len(poles), dtype=np.float32)
    dry_compound = np.zeros(len(poles), dtype=np.int32)

    for i, (idxs, intensity, month) in enumerate(
            zip(indices_list, raw["강도_abs"].values, raw["month"].values)):
        if len(idxs) == 0:
            continue
        count_all[idxs] += 1
        for idx in idxs:
            if intensity > max_int_all[idx]:
                max_int_all[idx] = intensity
        if int(month) in FIRE_SEASON_MONTHS:
            dry_compound[idxs] += 1
        if (i + 1) % 500000 == 0:
            print(f"  {i+1:,}/{len(raw):,} 처리 중...")

    poles = poles.copy()
    poles["lightning_count_2km"]         = count_all
    poles["lightning_max_intensity_2km"] = max_int_all
    poles["lightning_dry_compound"]      = dry_compound

    out = OUT_DIR / "poles_lightning.parquet"
    poles.to_parquet(out, index=False)
    print(f"[LIGHTNING] 저장: {out}  ({len(poles):,}개 전봇대)")
    return poles


if __name__ == "__main__":
    poles = pd.read_parquet(OUT_DIR / "poles_aws_score.parquet")
    run(poles)
