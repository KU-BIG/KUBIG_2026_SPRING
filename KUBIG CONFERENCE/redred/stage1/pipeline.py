"""
pipeline.py
===========
반도체 업황 예측 파이프라인 — 마스터 실행기.

실행 방법:
    python stage1/pipeline.py

단계별 실행:
    python stage1/s1_data.py     # 데이터 수집 + 피처 엔지니어링
    python stage1/s2_tune.py     # XGBoost Optuna 초기 튜닝 (RMSE 목적함수)
    python stage1/s3_select.py   # 피처 선택 (다중공선성 + 중요도 + RFE)
    python stage1/s4_optimize.py # XGBoost Bear 최적화 (AsymLoss 목적함수)
    python stage1/s5_evaluate.py # 최종 평가 + 시각화

예상 소요 시간: 약 25~35분 (Optuna 2회 × 50 trial)
재현 목표:
  CV RMSE ≈ 6.87, AsymLoss ≈ 6.90, Bear DirAcc ≈ 90.5%
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("s1_data.py",     "Step 1  데이터 수집 + 피처 엔지니어링"),
    ("s2_tune.py",     "Step 2  Optuna 하이퍼파라미터 튜닝 (RMSE 목적함수)"),
    ("s3_select.py",   "Step 3  피처 선택 (다중공선성 + 중요도 + RFE)"),
    ("s4_optimize.py", "Step 4  XGBoost Bear 최적화 (AsymLoss 목적함수)"),
    ("s5_evaluate.py", "Step 5  최종 평가 + 시각화"),
]


def main():
    print("=" * 64)
    print("  반도체 업황 예측 파이프라인")
    print("=" * 64)

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
    print("  모델:   outputs/models/best_xgboost_final.pkl")
    print("  지표:   outputs/metrics/final_cv_metrics.csv")
    print("  시각화: outputs/figures/")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
