"""grid_cluster.py - EPSG:5186 100m 격자 + Connected Components + Step4 판정"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import label
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUT_DIR

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False
    print("[WARN] pyproj 없음 - 좌표 근사 사용")

GRID = 100
X_MIN, X_MAX = 270_000, 650_000
Y_MIN, Y_MAX = 350_000, 610_000


def to_5186(lon, lat):
    if _HAS_PYPROJ:
        tf = Transformer.from_crs("EPSG:4326","EPSG:5186", always_xy=True)
        return tf.transform(lon, lat)
    lat0, lon0 = 37.5, 128.0
    cx, cy = (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2
    x = (lon-lon0)*111320*np.cos(np.radians(lat0)) + cx
    y = (lat-lat0)*110540 + cy
    return x, y


def run(poles, T_percentile=80, min_cluster_size=10, K=20):
    poles = poles.copy()
    x, y = to_5186(poles["lon"].values, poles["lat"].values)
    poles["x"] = x; poles["y"] = y
    poles["grid_col"] = ((x - X_MIN) / GRID).astype(int).clip(0)
    poles["grid_row"] = ((y - Y_MIN) / GRID).astype(int).clip(0)

    n_cols = int((X_MAX-X_MIN)/GRID)
    n_rows = int((Y_MAX-Y_MIN)/GRID)
    grid = np.zeros((n_rows, n_cols), dtype=np.int8)

    threshold = np.percentile(poles["adjusted_score"], T_percentile)
    danger_poles = poles[poles["adjusted_score"] >= threshold]
    dp = danger_poles[["grid_row", "grid_col"]].astype(int)
    valid_dp = dp["grid_row"].between(0, n_rows-1) & dp["grid_col"].between(0, n_cols-1)
    grid[dp.loc[valid_dp, "grid_row"].values, dp.loc[valid_dp, "grid_col"].values] = 1
    print(f"[GRID] 위험 격자: {grid.sum():,}셀  (threshold={threshold:.4f})")

    struct = np.ones((3,3), dtype=int)
    labeled, n_cl = label(grid, structure=struct)
    danger_lbls = {l for l in range(1, n_cl+1) if (labeled==l).sum() >= min_cluster_size}
    danger_grid = np.isin(labeled, list(danger_lbls)).astype(np.int8)
    print(f"[GRID] 전체 클러스터: {n_cl}, danger(≥{min_cluster_size}셀): {len(danger_lbls)}")

    p_rows = poles["grid_row"].astype(int).clip(0, n_rows-1).values
    p_cols = poles["grid_col"].astype(int).clip(0, n_cols-1).values
    valid_p = (poles["grid_row"].between(0, n_rows-1) & poles["grid_col"].between(0, n_cols-1)).values
    in_danger_arr = np.zeros(len(poles), dtype=bool)
    in_danger_arr[valid_p] = danger_grid[p_rows[valid_p], p_cols[valid_p]].astype(bool)
    poles["in_danger_cluster"] = in_danger_arr

    score_thr = poles["adjusted_score"].quantile(1 - K/100)
    poles["decision"] = (
        poles["in_danger_cluster"] & (poles["adjusted_score"] >= score_thr)
    ).astype(int)

    n_dec = poles["decision"].sum()
    print(f"[GRID] decision=1: {n_dec:,}개 ({n_dec/len(poles)*100:.2f}%)")
    poles.to_parquet(OUT_DIR / "poles_decision.parquet", index=False)
    return poles


if __name__ == "__main__":
    poles = pd.read_parquet(OUT_DIR / "poles_lightning.parquet")
    run(poles)
