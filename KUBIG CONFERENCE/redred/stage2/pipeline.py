"""
pipeline.py
===========
SK하이닉스 주가 예측 파이프라인 (Stage 2) — 마스터 실행기.

실행 방법:
    python stage2/pipeline.py

단계별 실행:
    python stage2/s1_dates.py       # 분기 날짜 생성 + 타겟 수익률 산출
    python stage2/s2_data.py        # A~E 피처 수집
    python stage2/s3_stage1_feat.py # Stage 1 expanding window pseudo 예측
    python stage2/s4_features.py    # 피처 엔지니어링 + 선택
    python stage2/s5_tune.py        # XGBoost Optuna 튜닝
    python stage2/s6_evaluate.py    # 최종 평가 + 시각화

선행 조건:
    Stage 1 파이프라인이 완료되어 다음 파일이 존재해야 함:
      stage1/outputs/data/features_dataset.csv
      stage1/outputs/models/best_xgboost_final.pkl

예상 소요 시간: 약 30~50분
  (s3 expanding window ~100회 재훈련 + s5 Optuna 2×50 trial)
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Stage 1 선행 조건 체크
STAGE1_DEPS = [
    os.path.join(ROOT_DIR, "stage1", "outputs", "data",   "features_dataset.csv"),
    os.path.join(ROOT_DIR, "stage1", "outputs", "models", "best_xgboost_final.pkl"),
]

STEPS = [
    ("s1_dates.py",       "Step 1  분기 날짜 생성 + 타겟 수익률 산출"),
    ("s2_data.py",        "Step 2  A~E 피처 수집 (yfinance · FRED · WSTS)"),
    ("s3_stage1_feat.py", "Step 3  Stage 1 Expanding Window Pseudo 예측"),
    ("s4_features.py",    "Step 4  피처 엔지니어링 + 피처 선택"),
    ("s5_tune.py",        "Step 5  XGBoost Optuna 튜닝 (RMSE → AsymLoss)"),
    ("s6_evaluate.py",    "Step 6  최종 평가 + 시각화"),
]


def check_stage1():
    missing = [p for p in STAGE1_DEPS if not os.path.exists(p)]
    if missing:
        print("[오류] Stage 1 파이프라인 출력 파일이 없습니다:")
        for p in missing:
            print(f"  ✗ {p}")
        print("\n  먼저 Stage 1을 실행하세요:  python stage1/pipeline.py")
        sys.exit(1)
    print("[확인] Stage 1 출력 파일 존재 ✓")


def main():
    print("=" * 64)
    print("  SK하이닉스 주가 예측 파이프라인  (Stage 2)")
    print("=" * 64)

    check_stage1()

    for script, desc in STEPS:
        print(f"\n{'─'*64}")
        print(f"  {desc}")
        print(f"{'─'*64}")
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, script)],
            cwd=BASE_DIR,
        )
        if result.returncode != 0:
            print(f"\n[오류] {script} 실패 — 파이프라인 중단.")
            sys.exit(1)

    print(f"\n{'='*64}")
    print("  파이프라인 완료.")
    print()
    print("  모델:   stage2/outputs/models/skh_xgb_final.pkl")
    print("  지표:   stage2/outputs/metrics/final_cv_metrics.csv")
    print("  시각화: stage2/outputs/figures/")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
