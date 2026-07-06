"""공통 경로·상수. stage2 모든 스크립트에서 import."""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # stage2/
ROOT_DIR   = os.path.dirname(BASE_DIR)                     # 프로젝트 루트

DATA_DIR   = os.path.join(BASE_DIR, "outputs", "data")
MODEL_DIR  = os.path.join(BASE_DIR, "outputs", "models")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

# Stage 1 경로
STAGE1_FEATURES_PATH = os.path.join(ROOT_DIR, "stage1", "outputs", "data", "features_dataset.csv")
STAGE1_FINAL_PKL     = os.path.join(ROOT_DIR, "stage1", "outputs", "models", "best_xgboost_final.pkl")
WSTS_PATH            = os.path.join(ROOT_DIR, "wsts_historical.xlsx")

# Stage 2 데이터 경로
DATES_PATH    = os.path.join(DATA_DIR, "quarterly_dates.csv")
RAW_PATH      = os.path.join(DATA_DIR, "raw_quarterly.csv")
S1PRED_PATH   = os.path.join(DATA_DIR, "stage1_predictions.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "stage2_features.csv")
TUNED_PKL     = os.path.join(MODEL_DIR, "skh_xgb_tuned.pkl")
FINAL_PKL     = os.path.join(MODEL_DIR, "skh_xgb_final.pkl")

# FRED API 키 (환경변수에서 로드)
FRED_API_KEY = os.getenv("FRED_API_KEY")

# 날짜 범위
START_YEAR = 2000
END_YEAR   = 2026

# 타겟: SK하이닉스 6개월 종가 수익률 (%)
PRIMARY_TARGET = "TARGET_SKH_6M_RET"

# CV 설정 (분기 단위)
TEST_EVAL_SIZE = 20   # hold-out 분기 수 (5년)
N_SPLITS       = 5
TEST_SIZE      = 4    # fold당 test 분기
MIN_TRAIN      = 20   # fold 최소 train 분기
N_TRIALS       = 50
RANDOM_STATE   = 42

# Asymmetric Loss 가중치 (Bear 오예측 페널티 강화)
W_BULL_CORRECT = 1.0
W_BULL_WRONG   = 2.0
W_BEAR_CORRECT = 1.5
W_BEAR_WRONG   = 3.0
BEAR_SAMPLE_W  = 2.0

for d in [DATA_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)
