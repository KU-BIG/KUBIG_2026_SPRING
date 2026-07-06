# 분석 프로그램 코드 — 기상 데이터 및 공간정보 기반 전력설비 인근 화재 위험도 분석

강원도 전력설비(전주) 인근 산불 위험도를 비지도 이상탐지(Isolation Forest) 기반으로
예측하는 파이프라인 코드입니다. F2 score 기준으로 최적화되었습니다.

## 디렉터리 구조

```
kangwon_fire/                  # 메인 분석 패키지
├── config.py                  # 경로·피처·시즌(11~5월) 설정
├── main.py                    # ★ 전체 파이프라인 실행 진입점 (7단계)
├── preprocessing/
│   ├── aws_prep.py            # AWS(방재기상관측) 전처리
│   ├── asos_prep.py           # ASOS(종관기상관측) 전처리
│   ├── fwi_prep.py            # FWI(산불기상지수) 계산
│   ├── lightning_prep.py      # 낙뢰 데이터 집계 (전주 2km 반경)
│   └── merge.py               # AWS·ASOS 병합 + 화재시즌 필터링
├── model/
│   ├── isolation_forest.py    # Isolation Forest 이상탐지 (관측지점별 anomaly score)
│   └── grid_cluster.py        # 격자 클러스터링 + 최종 이진 판정(decision)
└── validation/
    └── wildfire_eval.py       # 과거 산불 신고 데이터 기반 precision/recall/F2 검증

terrain_prep.py                # DEM(고도·경사·향) 기반 지형 가중치 산출
run_gridsearch.py              # 하이퍼파라미터 그리드서치
run_eval.py                    # 검증 단독 실행
visualize.py                   # 결과 지도·대시보드 시각화
eda/eda.py                     # 탐색적 데이터 분석
```

## 실행 방법

```bash
python kangwon_fire/main.py
```

## 파이프라인 단계 (main.py)

1. AWS 전처리
2. ASOS 전처리
3. FWI 산불기상지수 계산
4. AWS·ASOS 병합 + 화재시즌(11~5월) 필터
5. Isolation Forest 이상탐지 → 관측지점별 anomaly score
6. 낙뢰 집계 (전주 반경 2km)
7. 격자 클러스터링 + 지형 가중치 반영 → 최종 이진 판정(decision: 0/1)

## 참고

- 입력 데이터(기상 원자료, 전주 좌표, DEM 지형변수 등)는 용량 관계로 코드에 미포함입니다.
  `config.py`의 경로 설정을 참고하세요.
- 최종 결과물은 `pole_id, lon, lat, decision` 컬럼의 CSV로 출력됩니다.
