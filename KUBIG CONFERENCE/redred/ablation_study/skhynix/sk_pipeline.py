"""
sk_pipeline.py — SK하이닉스 주가 수익률 Ablation Study 마스터 실행기
=====================================================================
Step 0 ~ Step 3을 순차적으로 실행한다.

사용법:
  python skhynix/sk_pipeline.py           # 전체 실행
  python skhynix/sk_pipeline.py --from 2  # Step 2부터 재실행

Step 0: 공급 모델 OOS 예측값 추출 (약 2-4분 소요)
Step 1: SK하이닉스 / KOSPI 월별 주가 수집
Step 2: 월별 피처 매트릭스 빌드
Step 3: Ablation Study (Walk-Forward CV + SHAP)
"""

import argparse
import time

STEPS = [
    (0, "sk0_oos",       "공급 모델 OOS 예측값 추출"),
    (1, "sk1_price",     "SK하이닉스 / KOSPI 주가 수집"),
    (2, "sk2_features",  "월별 피처 매트릭스 빌드"),
    (3, "sk3_ablation",  "Ablation Study"),
]


def run_from(start_step: int):
    total_start = time.time()
    for step_num, module_name, desc in STEPS:
        if step_num < start_step:
            print(f"  [Step {step_num}] 스킵 ({desc})")
            continue
        print(f"\n{'#'*64}")
        print(f"  Step {step_num}: {desc}")
        print(f"{'#'*64}")
        t0 = time.time()
        mod = __import__(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  Step {step_num} 소요시간: {elapsed:.1f}초")

    total = time.time() - total_start
    print(f"\n{'='*64}")
    print(f"  파이프라인 완료. 총 소요시간: {total:.1f}초")
    print(f"  결과 위치:")
    print(f"    skhynix/outputs/data/        — parquet 파일")
    print(f"    skhynix/outputs/figures/     — 시각화 (SHAP, 비교 차트)")
    print(f"    skhynix/outputs/metrics/     — sk_ablation_results.csv")
    print(f"{'='*64}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SK하이닉스 주가 수익률 Ablation Study 파이프라인")
    parser.add_argument("--from", dest="from_step", type=int, default=0,
                        help="시작할 Step 번호 (0=전체, 1=Step1부터, ...)")
    args = parser.parse_args()
    run_from(args.from_step)
