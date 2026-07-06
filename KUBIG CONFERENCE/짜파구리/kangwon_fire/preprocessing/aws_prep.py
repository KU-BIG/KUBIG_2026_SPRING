import glob, sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AWS_DIR, AWS_META, OUT_DIR


def load_aws_meta():
    meta = pd.read_csv(AWS_META, encoding="cp949")
    meta = (meta.sort_values("시작일", ascending=False)
                .drop_duplicates("지점", keep="first")
                [["지점", "위도", "경도", "지점명"]]
                .reset_index(drop=True))
    return meta


def load_aws_raw():
    files = sorted(glob.glob(str(AWS_DIR / "SURFACE_AWS_*.csv")))
    dfs = [pd.read_csv(f, encoding="cp949") for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    raw["일시"] = pd.to_datetime(raw["일시"])
    raw = raw.sort_values(["지점", "일시"]).drop_duplicates(["지점", "일시"])
    print(f"  AWS raw: {len(raw):,}행, {raw['지점'].nunique()}개 지점")
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


def _interpolate(df):
    cols = ["평균기온(°C)", "최저기온(°C)", "최고기온(°C)",
            "일강수량(mm)", "최대 순간 풍속(m/s)", "평균 풍속(m/s)",
            "최대 순간 풍속 풍향(deg)"]
    for col in cols:
        if col in df.columns:
            df[col] = df.groupby("지점")[col].transform(
                lambda s: s.interpolate("linear").ffill().bfill())
    df["일강수량(mm)"] = df["일강수량(mm)"].clip(lower=0)
    return df


def _add_derived(df):
    df["temp_range"] = df["최고기온(°C)"] - df["최저기온(°C)"]
    df["_dry"] = (df["일강수량(mm)"] < 1).astype(int)
    def streak(s):
        r, c = [], 0
        for v in s:
            c = c + 1 if v else 0
            r.append(c)
        return r
    df["drought_days"] = df.groupby("지점")["_dry"].transform(streak)
    df.drop(columns=["_dry"], inplace=True)
    df["wind_ma3"] = df.groupby("지점")["평균 풍속(m/s)"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    wd = np.radians(df["최대 순간 풍속 풍향(deg)"])
    df["wind_sin"] = np.sin(wd)
    df["wind_cos"] = np.cos(wd)
    m = df["일시"].dt.month
    df["month_sin"] = np.sin(m * 2 * np.pi / 12)
    df["month_cos"] = np.cos(m * 2 * np.pi / 12)
    return df


def run():
    print("[AWS] 메타 로드...")
    meta = load_aws_meta()
    print(f"  지점 수: {len(meta)}")
    print("[AWS] 원시 데이터 로드...")
    raw = load_aws_raw()
    print("[AWS] 날짜 완성 + 보간...")
    df = _fill_dates(raw)
    df = _interpolate(df)
    print("[AWS] 파생 피처...")
    df = _add_derived(df)
    out = OUT_DIR / "aws_processed.parquet"
    df.to_parquet(out, index=False)
    print(f"[AWS] 저장: {out}  ({len(df):,}행)")
    return df, meta


if __name__ == "__main__":
    run()
