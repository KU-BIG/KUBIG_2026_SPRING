"""
wildfire_eval.py — 4순위 (Validation)

전봇대 데이터가 강원도 일부만 커버하므로 화재 중심(fire-centric) 메트릭 사용:
  - fire_recall  : 커버리지 내 화재 중 decision=1 전봇대가 반경 R 이내에 있는 비율
  - fire_precision: decision=1 전봇대 중 반경 R 이내에 화재가 있는 전봇대 비율
  - F2           : recall에 2배 가중 (조기경보 특성)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree
from itertools import product
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WILDFIRE_CSV, OUT_DIR, FIRE_SEASON_MONTHS

COVERAGE_BUFFER_KM = 10  # 전봇대 bbox 외곽 버퍼
WILDFIRE_PLACE_TYPES = ["산불", "들불", "야외"]  # 산불 관련 발생장소만 평가


def to_decimal(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return np.nan
    if v > 1000:
        d = int(v // 10000); m = int((v % 10000) // 100); s = v % 100
        return d + m / 60 + s / 3600
    return v


def load_wildfire():
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(WILDFIRE_CSV, encoding=enc,
                             on_bad_lines="skip", low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"화재 CSV 로드 실패: {WILDFIRE_CSV}")

    df = df[df["GRNDS_CTPV_NM"].isin(["강원도", "강원특별자치도"])].copy()
    df = df.dropna(subset=["ACDNT_OCRN_LAT", "ACDNT_OCRN_LOT"])
    df["fire_lat"] = df["ACDNT_OCRN_LAT"].apply(to_decimal)
    df["fire_lon"] = df["ACDNT_OCRN_LOT"].apply(to_decimal)
    df = df.dropna(subset=["fire_lat", "fire_lon"])
    df["DCLR_YMD"] = pd.to_datetime(
        df["DCLR_YMD"].astype(str), format="%Y%m%d", errors="coerce")
    df["month"] = df["DCLR_YMD"].dt.month
    df = df[df["month"].isin(FIRE_SEASON_MONTHS)]
    # 산불/들불/야외만 (건물화재·차량화재 제외)
    if "ACDNT_OCRN_PLC_NM" in df.columns:
        df = df[df["ACDNT_OCRN_PLC_NM"].isin(WILDFIRE_PLACE_TYPES)]
    # 강원도 좌표 유효 범위
    df = df[df["fire_lat"].between(37.0, 39.0) &
            df["fire_lon"].between(127.0, 130.5)]
    print(f"[EVAL] 강원 11~5월 산불/들불/야외: {len(df):,}건")
    return df.reset_index(drop=True)


def _pole_coverage_bbox(poles: pd.DataFrame, buffer_km: float = COVERAGE_BUFFER_KM):
    """전봇대 bounding box + buffer (위도/경도 도 단위 근사)"""
    buf_deg = buffer_km / 111.0
    return {
        "lat_min": poles["lat"].min() - buf_deg,
        "lat_max": poles["lat"].max() + buf_deg,
        "lon_min": poles["lon"].min() - buf_deg / np.cos(np.radians(poles["lat"].mean())),
        "lon_max": poles["lon"].max() + buf_deg / np.cos(np.radians(poles["lat"].mean())),
    }


def eval_metrics(poles: pd.DataFrame, fires: pd.DataFrame,
                 radius_m: int = 3000) -> tuple:
    """
    화재 중심 메트릭 (전봇대 커버리지 내 화재만 평가 대상)

    fire_recall  : 커버리지 내 화재 중 R 이내 decision=1 전봇대 있는 비율
    fire_precision: decision=1 전봇대 중 R 이내 화재 있는 비율
    F2           : beta=2 (recall 2배 가중)
    """
    # 커버리지 내 화재만 필터
    bbox = _pole_coverage_bbox(poles)
    fires_in = fires[
        fires["fire_lat"].between(bbox["lat_min"], bbox["lat_max"]) &
        fires["fire_lon"].between(bbox["lon_min"], bbox["lon_max"])
    ].copy()
    n_fires_total = len(fires)
    n_fires_in    = len(fires_in)

    if n_fires_in == 0:
        print(f"  [WARN] 전봇대 커버리지({COVERAGE_BUFFER_KM}km 버퍼) 내 화재 없음")
        return 0.0, 0.0, 0.0

    r_rad = (radius_m / 1000) / 6371.0

    # ── fire_recall: 화재 → decision=1 전봇대 탐색 ─────────────────
    danger_poles = poles[poles["decision"] == 1]
    if len(danger_poles) == 0:
        return 0.0, 0.0, 0.0

    dp_c    = np.radians(danger_poles[["lat", "lon"]].values)
    fire_c  = np.radians(fires_in[["fire_lat", "fire_lon"]].values)
    dp_tree = BallTree(dp_c, metric="haversine")

    counts = dp_tree.query_radius(fire_c, r=r_rad, count_only=True)
    fires_detected = int((counts > 0).sum())
    fire_recall = fires_detected / n_fires_in

    # ── fire_precision: decision=1 전봇대 → 화재 탐색 ──────────────
    all_fire_c  = np.radians(fires[["fire_lat", "fire_lon"]].values)
    fire_tree   = BallTree(all_fire_c, metric="haversine")
    counts_pole = fire_tree.query_radius(dp_c, r=r_rad, count_only=True)
    poles_near_fire = int((counts_pole > 0).sum())
    fire_precision  = poles_near_fire / len(danger_poles)

    # ── F2 ──────────────────────────────────────────────────────────
    beta = 2
    if fire_precision + fire_recall > 0:
        f2 = ((1 + beta**2) * fire_precision * fire_recall /
              (beta**2 * fire_precision + fire_recall))
    else:
        f2 = 0.0

    print(f"  커버리지 내 화재: {n_fires_in}/{n_fires_total} "
          f"(전봇대 bbox ±{COVERAGE_BUFFER_KM}km)")
    print(f"  감지된 화재: {fires_detected}/{n_fires_in} "
          f"(반경 {radius_m}m 내 decision=1)")
    print(f"  danger 전봇대 수: {len(danger_poles):,}")

    return round(fire_precision, 4), round(fire_recall, 4), round(f2, 4)


def grid_search(poles_base: pd.DataFrame, fires: pd.DataFrame,
                T_range, n_range, K_range, radius_m: int = 3000) -> tuple:
    from model.grid_cluster import run as cluster_run
    best = {"f2": -1, "params": {}}
    results = []
    total = len(list(T_range)) * len(n_range) * len(list(K_range))
    print(f"[EVAL-GS] 총 조합: {total}개")

    for i, (T, n, K) in enumerate(product(T_range, n_range, K_range), 1):
        poles = cluster_run(poles_base.copy(), T, n, K)
        p, r, f2 = eval_metrics(poles, fires, radius_m)
        results.append({"T": T, "n": n, "K": K,
                        "precision": p, "recall": r, "f2": f2})
        if f2 > best["f2"]:
            best = {"f2": f2, "params": {"T": T, "n": n, "K": K}}
        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] best f2={best['f2']:.4f} @ {best['params']}")

    res_df = pd.DataFrame(results).sort_values("f2", ascending=False)
    res_df.to_csv(OUT_DIR / "grid_search_results.csv", index=False)
    print(f"[EVAL-GS] 최적: {best}")
    return best, res_df


def run(poles: pd.DataFrame, radius_m: int = 3000) -> tuple:
    fires = load_wildfire()
    p, r, f2 = eval_metrics(poles, fires, radius_m)
    print(f"[EVAL] radius={radius_m}m  P={p:.4f}  R={r:.4f}  F2={f2:.4f}")
    ok_p = "✓" if p >= 0.20 else "✗"
    ok_r = "✓" if r >= 0.40 else "✗"
    ok_f = "✓" if f2 >= 0.30 else "✗"
    print(f"       목표: P≥0.20{ok_p}  R≥0.40{ok_r}  F2≥0.30{ok_f}")
    return p, r, f2


if __name__ == "__main__":
    poles = pd.read_parquet(OUT_DIR / "poles_decision.parquet")
    run(poles)
