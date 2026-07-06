   # Ablation Study — 반도체 공급 신호의 주가 선행성 검증

   ## 개요

   이 디렉토리는 Stage 1에서 생성한 반도체 업황 예측값(`wsts_pred_t6`)이 실제 반도체 기업 주가 수익률의 유의미한 선행 신호인지를 검증하기 위한 Ablation Study 코드와 결과 문서를 담고 있다.

   분석 시점: 2026-05-28 기준 Stage 1 스냅샷 사용
   (`ablation_study/model/outputs/`에 보존된 해당 시점 산출물 기준으로 수행되었으며, 현재 루트 `stage1/outputs/`와는 별개임)

   ## 분석 구조

   Model A/B/C 비교로 공급 신호의 기여도를 정량화:
   - **A (Full)**: 공급 신호 + 거시경제 + 종목 전용 피처
   - **B (No supply)**: 공급 신호 제외
   - **C (Supply only)**: 공급 신호 단독

   대상 종목: SK하이닉스, ASML
   분석 기간: 2004-12 ~ 2023-07 (월별, 224개월)
   검증 방식: Expanding-window Walk-forward CV

   상세 분석 결과는 [`docs/stage2_summary.md`](docs/stage2_summary.md) 참고.

   ## 디렉토리 구조

   ```
   ablation_study/
   ├── skhynix/    SK하이닉스 ablation 파이프라인 (sk0~sk4)
   ├── asml/       ASML ablation 파이프라인 (asml1~asml3)
   ├── model/      Stage1 스냅샷 산출물 보관 (outputs/만 존재, 코드 없음)
   └── docs/       분석 결과 보고서 (stage2_summary.md)
   ```

   ## 실행 방법

   ### 사전 조건

   1. 루트 `requirements.txt` 설치 완료
   2. `FRED_API_KEY` 환경변수 설정
   3. `ablation_study/model/outputs/`에 Stage 1 스냅샷 산출물 존재 확인
      (git에는 올라가지 않으므로 최초 실행 시 `stage1/pipeline.py`로 생성한 후 `ablation_study/model/outputs/`에 복사 필요)
      필요한 파일:
      ```
      ablation_study/model/outputs/data/features_dataset.csv
      ablation_study/model/outputs/models/best_xgboost_final.pkl
      ```

   ### SK하이닉스 분석 실행 순서

   ```bash
   cd ablation_study/skhynix

   python sk0_oos.py       # Step 0: 공급 모델 OOS 예측값 추출 (leakage 방지)
   python sk1_price.py     # Step 1: SK하이닉스 / KOSPI 월별 주가 수집
   python sk2_features.py  # Step 2: 월별 피처 매트릭스 빌드
   python sk3_ablation.py  # Step 3: Model A/B/C ablation 실험 (Walk-Forward CV + SHAP)
   python sk4_horizon.py   # Horizon sensitivity 분석 (h=1~12, 별도 실행)
   ```

   > Step 0~3은 `python sk_pipeline.py`로 한 번에 실행 가능 (`--from N`으로 특정 단계부터 재실행 가능). `sk4_horizon.py`는 파이프라인에 포함되지 않은 별도 스크립트로, Step 0~3 완료 후 수동 실행한다.

   ### ASML 분석 실행 순서

   ```bash
   cd ablation_study/asml

   python asml1_features.py   # Step 1: 피처 수집 및 병합 (EUR 환율, SOX 포함)
   python asml2_ablation.py   # Step 2: Model A/B/C ablation 실험
   python asml3_horizon.py    # Horizon sensitivity 분석 (별도 실행)
   ```

   > Step 1~2는 `python asml_pipeline.py`로 한 번에 실행 가능. `asml3_horizon.py`는 파이프라인에 포함되지 않은 별도 스크립트다.

   ### 주의사항

   - 각 스크립트는 순서대로 실행해야 합니다 (이전 스크립트의 outputs에 의존).
   - `outputs/` 디렉토리는 `.gitignore`로 제외되어 있으므로 git clone 후에는 로컬에서 직접 생성해야 합니다.
   - `sk_ablation_results_v2_deprecated.csv`는 이전 버전 결과입니다 (문서가 인용하는 수치는 `sk_ablation_results.csv` 기준).

   ## 핵심 결과 요약

   상세 내용은 [`docs/stage2_summary.md`](docs/stage2_summary.md) 참고.

   | 종목 | Model A DirAcc | Δ DirAcc (A-B) | SHAP 순위 |
   |------|---------------|----------------|-----------|
   | SK하이닉스 | 75.6% | +4.3%p | 2위 / 12개 |
   | ASML | 79.3% | +1.2%p | 11위 / 15개 |

   공급 신호(`wsts_pred_t6`)는 두 종목 모두에서 RMSE 감소 + DirAcc 증가를 동시에 달성했으며, h=3~7 구간에서 기여가 집중됨을 확인했다 (h=6 설계 결정 사후 검증).
