# 프로젝트 개요
이 프로젝트는 FNSPID 금융 뉴스/주가 데이터셋을 사용해, 기존 감성(`Scaled_sentiment`)을 그대로 쓰는 베이스라인과 새로 설계한 가중 감성(`News_flag` + 거래량 z-score 기반)을 비교하는 CNN 시계열 예측 실험입니다.

# 기존 데이터셋과의 차이점
- 기존 데이터: `Scaled_sentiment` 컬럼을 원본 그대로 사용
- 변경 데이터: `Scaled_sentiment`를 아래 방식으로 재계산

```python
sent_center = (Sentiment_gpt - 3) / 2  # 1~5 -> -1~1
vol_z = zscore(Volume, rolling window=N)
intensity = News_flag * (1 + k * vol_z)
intensity = clip(intensity, lower=0)
Scaled_sentiment = sent_center * intensity
```

즉, 뉴스 존재 여부(`News_flag`)와 거래량 급증(관심도)을 감성 강도에 반영하도록 설계했습니다.

# 입력 CSV 형식 (모델에 넣을 데이터)
실험용 CSV는 **종목당 1개 파일**, 파일명은 `{종목심볼}.csv` (예: `KO.csv`, `AMD.csv`)로 두고, 각 모델 폴더의 **`data/`** 디렉터리에 넣습니다.

## 필수 컬럼 (CNN / RNN / GRU / LSTM / TimesNet)
| 컬럼명 | 설명 | 비고 |
|--------|------|------|
| **Close** | 종가 | 예측 타깃. 반드시 존재 |
| **Volume** | 거래량 | 정규화 대상(0번 컬럼과 함께) |
| **Scaled_sentiment** | 스케일된 감성 (0~1 등) | sentiment 실험 시 필수, nonsentiment 시에는 사용 안 함 |

- **Sentiment 실험**: CSV에 `Close`, `Volume`, `Scaled_sentiment` 세 컬럼이 있어야 함 (`sentiment_config.json`의 `columns`).
- **Nonsentiment 실험**: `Close`, `Volume` 두 컬럼만 있으면 됨 (`nonsentiment_config.json`의 `columns`).
- **행 순서**: 반드시 **시간 오름차순**(과거 → 미래). 코드에서 `train_test_split` 비율로 앞부분=학습, 뒷부분=테스트로 잘라 쓰므로 shuffle 하면 안 됨.

## 선택 컬럼 (Transformer)
Transformer 모델은 다음 6개 컬럼 사용: `Volume`, `Open`, `High`, `Low`, `Close`, `Scaled_sentiment`.  
CNN/LSTM용 CSV에 **Open, High, Low**를 추가해 두면 Transformer 실험에 그대로 쓸 수 있음.

## 권장: Date 컬럼
코드에서 직접 사용하지는 않지만 **Date** 컬럼이 있으면 시간 순서 검증·디버깅에 유리함.  
전처리 파이프라인(`data_processor/price_news_integrate.py`) 결과물에는 보통 `Date`, `Close`, `Volume`, `Scaled_sentiment`, `News_flag` 등이 포함됨.

## CSV 예시 (최소 구성 – sentiment)
```csv
Date,Close,Volume,Scaled_sentiment
1999-01-04,25.5,1200000,0.45
1999-01-05,25.8,1150000,0.52
...
```
- **데이터 수**: 학습/테스트 분할과 시퀀스 길이(예: 50)를 고려해 **최소 약 333일 이상** 권장(전처리 코드에서 333 미만이면 스킵하는 경우 있음).

## 데이터 로딩 모듈 (dataset.py)
프로젝트 루트의 `dataset.py`는 CSV 로드 → **날짜 기준 정렬·분할** → 시계열 윈도우 생성을 한 곳에서 처리한다.  
기존 `core/data_processor.DataLoader`와 인자를 맞춰 두었으므로, `val_ratio=0`이면 동일하게 쓸 수 있다.

- **날짜 기준 split**: CSV에 `Date` 컬럼이 있으면 자동으로 오름차순 정렬 후 비율로 train/val/test 분할 (shuffle 없음).
- **검증 구간**: `FNSPIDDataset(..., val_ratio=0.1)` 처럼 주면 `get_val_data(seq_len, normalise)`로 검증용 (x, y)를 쓸 수 있다.
- **분할 검증**: `get_split_bounds()`로 train/val/test 구간의 인덱스·날짜를 확인할 수 있다.

```python
# 프로젝트 루트 또는 PYTHONPATH에 FNSPID가 있을 때
from dataset import FNSPIDDataset  # 또는 from dataset import DataLoader

data = FNSPIDDataset(
    "data/KO.csv",
    split=0.85,
    cols=["Close", "Volume", "Scaled_sentiment"],
    cols_to_norm=[0, 1],
    pred_len=3,
    val_ratio=0,       # 0이면 기존과 동일 (train 85%, test 15%)
    date_column="Date",
)
x_train, y_train = data.get_train_data(seq_len=50, normalise=True)
x_test, y_test, y_base = data.get_test_data(seq_len=50, normalise=True, cols_to_norm=[0, 1])
# 검증 사용 시: x_val, y_val = data.get_val_data(seq_len=50, normalise=True)
```

---

# 실험 요약 (CNN / sentiment 설정)
## 1. 데이터 정제/가공 방식
- 원본 폴더: `dataset_test/CNN-for-Time-Series-Prediction/data`
- 파생 폴더 생성:
  - `data_wN10_k0p1`, `data_wN10_k0p3`, `data_wN10_k0p6`
  - `data_wN20_k0p1`, `data_wN20_k0p3`, `data_wN20_k0p6`
  - `data_wN60_k0p1`, `data_wN60_k0p3`, `data_wN60_k0p6` (실험은 아직 미진행)
- 각 CSV에 아래 처리 적용:
  - 기존 `Scaled_sentiment` 보존: `Scaled_sentiment_orig`
  - `Scaled_sentiment` 재계산: 위 식 적용

## 2. 실험 설정
- 모델: `CNN-for-Time-Series-Prediction`
- 데이터 분할: `train_test_split = 0.85`
- 시퀀스 길이: 50
- 예측 길이: 3
- 비교 기준: 원본 `data` 폴더의 sentiment 결과 vs 각 가중 폴더 결과

## 3. 원본 데이터(CNN sentiment) 결과 (기존 데이터)
- 경로: `test_result_5/*_sentiment_2024013123`

- GOOG: MAE 0.052328, MSE 0.003694, R2 0.194527
- TSM: MAE 0.088045, MSE 0.011136, R2 0.516412
- WMT: MAE 0.024453, MSE 0.000930, R2 0.351031

## 4. 가중 데이터 실험 결과 요약 (원본 대비 변화량)
### A) `data_wN10_k0p1`
- KO: dMAE -0.002338, dMSE -0.000263, dR2 +0.059120
- TSM: dMAE -0.007757, dMSE +0.000079, dR2 -0.003431
- GOOG: dMAE -0.005582, dMSE -0.000457, dR2 +0.099685
- WMT: dMAE +0.013908, dMSE +0.001205, dR2 -0.841204
- 요약: KO/GOOG 개선, WMT 크게 악화

### B) `data_wN10_k0p3`
- KO: dMAE -0.000595, dMSE -0.000064, dR2 +0.014427
- TSM: dMAE -0.017995, dMSE -0.002848, dR2 +0.123680
- GOOG: dMAE +0.000804, dMSE +0.000511, dR2 -0.111498
- WMT: dMAE +0.009560, dMSE +0.000660, dR2 -0.460894
- 요약: TSM 개선, KO 미세 개선, 나머지 악화

### C) `data_wN10_k0p6`
- KO: dMAE +0.001159, dMSE +0.000173, dR2 -0.039001
- TSM: dMAE -0.021169, dMSE -0.003241, dR2 +0.140738
- GOOG: dMAE -0.009605, dMSE -0.000932, dR2 +0.203152
- WMT: dMAE +0.017680, dMSE +0.001851, dR2 -1.292318
- 요약: TSM/GOOG 크게 개선, WMT 크게 악화

### D) `data_wN20_k0p1`
- KO: dMAE +0.006021, dMSE +0.000943, dR2 -0.212321
- TSM: dMAE -0.020875, dMSE -0.003310, dR2 +0.143747
- GOOG: dMAE -0.010450, dMSE -0.000909, dR2 +0.198251
- WMT: dMAE +0.016551, dMSE +0.001330, dR2 -0.928696
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화

### E) `data_wN20_k0p3`
- KO: dMAE +0.005938, dMSE +0.000718, dR2 -0.161778
- TSM: dMAE -0.019123, dMSE -0.003477, dR2 +0.150968
- GOOG: dMAE -0.009687, dMSE -0.000946, dR2 +0.206326
- WMT: dMAE +0.012675, dMSE +0.000949, dR2 -0.662232
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화

### F) `data_wN20_k0p6`
- KO: dMAE +0.000205, dMSE +0.000118, dR2 -0.026643
- TSM: dMAE -0.021497, dMSE -0.003821, dR2 +0.165917
- GOOG: dMAE -0.008117, dMSE -0.000478, dR2 +0.104322
- WMT: dMAE +0.006079, dMSE +0.000410, dR2 -0.286549
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화(다른 k보다 악화폭은 작음)

## 5. 관찰 요약
- TSM, GOOG는 대부분의 가중 조합에서 개선 경향
- AMD는 대부분 악화
- WMT는 모든 조합에서 큰 악화(특히 k=0.6에서 심각)

---

# Data/ETL 업데이트 요약 (강산)

Data/ETL 리드 역할로 진행한 업데이트 내용을 정리하였다. 민혁님이 구현한 전처리·실험 파이프라인을 기준으로, 데이터 로딩 모듈화와 시간(날짜) 기준 분할 검증, 문서 보강을 추가하였다.

## 1. dataset.py 추가

프로젝트 루트에 `dataset.py`를 추가하였다. CSV 로드 → 날짜 기준 정렬 → 비율에 따른 train/val/test 분할 → 시계열 윈도우 생성을 한 모듈에서 처리하며, shuffle을 사용하지 않으므로 시간 순서가 유지된다. 기존에는 CNN/LSTM/GRU/RNN 각 모델 폴더에 DataLoader가 중복되어 있었으나, 동일 인자로 호환되므로 필요 시 `DataLoader` 대신 `FNSPIDDataset`로 교체하면 된다. 검증 구간이 필요하면 `val_ratio`를 지정하고 `get_val_data()`를 사용하면 되며, 분할 구간 확인은 `get_split_bounds()`로 할 수 있다.

## 2. 본 문서 보강 내용

위쪽에 다음 두 섹션을 추가하였다.

- **입력 CSV 형식**: 모델 입력용 CSV의 필수·선택 컬럼, 행 순서(시간 오름차순, shuffle 금지), Transformer용 컬럼, Date 권장, 예시 및 최소 데이터 수 안내.
- **데이터 로딩 모듈 (dataset.py)**: 사용 방법 및 예시 코드(위 "데이터 로딩 모듈" 섹션 참고).

## 3. 기존 코드와의 관계

전처리(`data_processor/price_news_integrate.py`), 감성 감쇠·Scaled_sentiment·가중 감성 계산, CNN 실험 및 본 문서의 실험 요약은 민혁님 구현 분이다. 본인이 수정·추가한 부분은 `dataset.py` 신규 작성과 본 문서의 "입력 CSV 형식", "데이터 로딩 모듈" 섹션뿐이며, `run.py` 및 각 모델의 `core/data_processor.py`는 변경하지 않았다. 필요 시 해당 경로에서 `DataLoader`를 `dataset.FNSPIDDataset`로 교체하여 사용하면 된다.

## 4. 남은 작업

`data_processor/`에서 수행하는 처리(주가·뉴스 병합, 날짜 정렬, 결측 감성 감쇠, Scaled_sentiment 계산 등)를 "어떤 컬럼을 어떤 순서로 어떻게 처리하는지" 노션에 정리하는 작업이 남아 있다.

## 5. 참고 파일 위치

- 데이터 로딩 모듈: 프로젝트 루트 `dataset.py`
- 기존 데이터 로더(모델별): `dataset_test/CNN-for-Time-Series-Prediction/core/data_processor.py` 등
- 전처리(병합·감쇠·스케일): `data_processor/price_news_integrate.py`
