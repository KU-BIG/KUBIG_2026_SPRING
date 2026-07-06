"""
terrain_prep.py
전봇대별 DEM 파생 지형 위험 점수 계산

입력: 전주데이터_DEM경사향추가.csv
출력: 전주데이터_terrain_score.csv

지형 위험 지표 (모두 높을수록 위험):
  south_danger   : 남향일수록 높음 (일사량 많아 건조)
  west_danger    : 서향일수록 높음 (강원 양간지풍 오후 건조)
  aspect_danger  : south 60% + west 40% 가중 합산
  slope_norm     : 경사 정규화 (가파를수록 화재 확산 빠름)
  terrain_score  : slope 50% + aspect 50% 종합 (0~1)
  terrain_weight : Step 4 판정 시 anomaly score 가중치 (0.5~1.0)
"""

import pandas as pd
import numpy as np

INPUT  = "전주데이터_DEM경사향추가.csv"
OUTPUT = "전주데이터_terrain_score.csv"


def calc_terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. aspect 결측(평지 slope=0) → 중립값 처리 ──────────────────────
    flat_mask = df["aspect_1"].isnull()
    df["aspect_1"] = df["aspect_1"].fillna(180.0)   # 임시 채움 (중립 방향)

    # ── 2. 남향 위험도: 남(180°)=1.0, 북(0°/360°)=0.0 ──────────────────
    aspect_rad = np.radians(df["aspect_1"])
    south_danger = (1 + np.cos(aspect_rad - np.pi)) / 2

    # ── 3. 서향 위험도: 서(270°)=1.0, 동(90°)=0.0 ─────────────────────
    #    강원도 양간지풍(서→동) 영향으로 서향 사면 오후 건조 위험 추가
    west_danger = (1 + np.cos(aspect_rad - 1.5 * np.pi)) / 2

    # ── 4. 평지(slope=0)는 방향 위험 = 중립(0.5) ──────────────────────
    aspect_danger = 0.6 * south_danger + 0.4 * west_danger
    aspect_danger[flat_mask] = 0.5

    # ── 5. 경사 정규화 (0~1) ────────────────────────────────────────────
    slope_max = df["slope_1"].max()
    slope_norm = df["slope_1"] / slope_max

    # ── 6. 종합 지형 위험 점수 (0~1) ────────────────────────────────────
    terrain_score = 0.5 * slope_norm + 0.5 * aspect_danger

    # ── 7. Step4 가중치: 0.5(최소)~1.0(최대) ────────────────────────────
    #    anomaly_score가 완전히 억제되지 않도록 하한 0.5 보장
    terrain_weight = 0.5 + 0.5 * terrain_score

    df["south_danger"]   = south_danger.round(4)
    df["west_danger"]    = west_danger.round(4)
    df["aspect_danger"]  = aspect_danger.round(4)
    df["slope_norm"]     = slope_norm.round(4)
    df["terrain_score"]  = terrain_score.round(4)
    df["terrain_weight"] = terrain_weight.round(4)

    return df


def summarize(df: pd.DataFrame) -> None:
    print("=" * 55)
    print("지형 위험 점수 요약")
    print("=" * 55)
    cols = ["slope_norm", "aspect_danger", "terrain_score", "terrain_weight"]
    print(df[cols].describe().round(3).to_string())

    print("\n─ terrain_score 분위 분포 ─")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        v = df["terrain_score"].quantile(q)
        print(f"  {int(q*100):>3}%ile : {v:.3f}")

    # 향 방향별 위험 비율
    bins   = [0, 45, 135, 225, 315, 360]
    labels = ["북(0-45/315-360)", "동(45-135)", "남(135-225)", "서(225-315)", "북(315-360)"]
    aspect_orig = df["aspect_1"].copy()
    direction = pd.cut(aspect_orig, bins=[0,45,135,225,315,360],
                       labels=["북", "동", "남", "서", "북2"], right=False)
    print("\n─ 향 방향별 전봇대 수 ─")
    print(direction.value_counts().to_string())

    print("\n─ terrain_weight 상위 10개 전봇대 ─")
    top = df.nlargest(10, "terrain_weight")[
        ["pole_id","lon","lat","slope_1","aspect_1","terrain_score","terrain_weight"]
    ]
    print(top.to_string(index=False))
    print("=" * 55)


def main():
    print(f"[1/3] 데이터 로드: {INPUT}")
    df = pd.read_csv(INPUT, encoding="cp949")
    print(f"      전봇대 수: {len(df):,}개 | 컬럼: {df.columns.tolist()}")

    print("[2/3] 지형 피처 계산...")
    df_out = calc_terrain_features(df)

    print("[3/3] 결과 저장 및 요약")
    # WKT, fid 제외하고 저장 (파이프라인에 필요한 컬럼만)
    save_cols = ["pole_id","lon","lat","elev_1","slope_1","aspect_1",
                 "south_danger","west_danger","aspect_danger",
                 "slope_norm","terrain_score","terrain_weight"]
    df_out[save_cols].to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"      저장 완료: {OUTPUT}")

    summarize(df_out)


if __name__ == "__main__":
    main()
