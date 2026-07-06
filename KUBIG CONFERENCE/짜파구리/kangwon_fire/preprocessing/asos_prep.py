import glob, sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASOS_DIR, ASOS_META, ASOS_COLS, OUT_DIR


def load_asos_meta():
    meta = pd.read_csv(ASOS_META, encoding="cp949")
    meta = (meta.sort_values("시작일", ascending=False)
                .drop_duplicates("지점", keep="first")
                [["지점", "위도", "경도", "지점명"]]
                .reset_index(drop=True))
    return meta


def load_asos_raw():
    files = sorted(glob.glob(str(ASOS_DIR / "SURFACE_ASOS_*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949", usecols=lambda c: c in ASOS_COLS)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {Path(f).name}: {e}")
    raw = pd.concat(dfs, ignore_index=True)
    raw["일시"] = pd.to_datetime(raw["일시"])
    raw = raw.sort_values(["지점", "일시"]).drop_duplicates(["지점", "일시"])
    print(f"  ASOS raw: {len(raw):,}행, {raw['지점'].nunique()}개 지점")
    return raw


def _fill_dates(df):
    full = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    parts = []
    for stn in df["지점"].unique():
        sub = df[df["지점"] == stn].set_index("일시").reindex(full)
        sub["지점"] = stn
        sub.index.name = "일시"
        parts.append(sub.reset_index())
    return pd.concat(parts, ignore_index=True)


def _eff_humidity(s, r=0.7, lookback=7):
    w = np.array([(1 - r) * (r ** i) for i in range(lookback)])
    return s.rolling(lookback, min_periods=lookback).apply(
        lambda x: float(np.dot(w, x[::-1])), raw=True)


def streak(s):
    res, cnt = [], 0
    for v in s:
        cnt = cnt + 1 if v else 0
        res.append(cnt)
    return res


def _median_impute(df, col):
    df["_m"] = df["일시"].dt.month
    gmed = df.groupby(["지점", "_m"])[col].transform("median")
    df[col] = df[col].fillna(gmed).ffill().bfill()
    df.drop(columns=["_m"], inplace=True)
    return df


def run():
    print("[ASOS] 메타 로드...")
    meta = load_asos_meta()
    print(f"  지점 수: {len(meta)}")
    print("[ASOS] 원시 데이터 로드...")
    raw = load_asos_raw()
    print("[ASOS] 날짜 완성...")
    df = _fill_dates(raw)

    for col in ["최소 상대습도(%)", "평균 상대습도(%)"]:
        if col in df.columns:
            df[col] = df.groupby("지점")[col].transform(
                lambda s: s.interpolate("linear").ffill().bfill())

    print("[ASOS] 실효습도 + dryness_index...")
    df["effective_humidity"] = df.groupby("지점")["평균 상대습도(%)"].transform(_eff_humidity)
    df["effective_humidity"] = df.groupby("지점")["effective_humidity"].transform(
        lambda s: s.ffill().bfill())
    df["dryness_index"] = (100 - df["effective_humidity"]).clip(lower=0)

    df["dry_wind_flag"] = (df["effective_humidity"] <= 35).astype(int)
    df["dry_wind_streak"] = df.groupby("지점")["dry_wind_flag"].transform(streak)

    if "합계 일사량(MJ/m2)" in df.columns:
        df = _median_impute(df, "합계 일사량(MJ/m2)")
    if "일 최심신적설(cm)" in df.columns:
        df["일 최심신적설(cm)"] = df["일 최심신적설(cm)"].fillna(0)

    for col in ["합계 일조시간(hr)", "평균 해면기압(hPa)",
                "평균 지면온도(°C)", "합계 대형증발량(mm)", "일강수량(mm)"]:
        if col in df.columns:
            df[col] = df.groupby("지점")[col].transform(
                lambda s: s.interpolate("linear").ffill().bfill())
    if "일강수량(mm)" in df.columns:
        df["일강수량(mm)"] = df["일강수량(mm)"].clip(lower=0)

    out = OUT_DIR / "asos_processed.parquet"
    df.to_parquet(out, index=False)
    print(f"[ASOS] 저장: {out}  ({len(df):,}행)")
    return df, meta


if __name__ == "__main__":
    run()
