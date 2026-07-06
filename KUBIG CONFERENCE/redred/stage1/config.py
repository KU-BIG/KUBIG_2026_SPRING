"""공통 경로·상수. 모든 sX 스크립트에서 import해 사용."""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # stage1/
ROOT_DIR   = os.path.dirname(BASE_DIR)                     # 프로젝트 루트

DATA_DIR   = os.path.join(BASE_DIR, "outputs", "data")
MODEL_DIR  = os.path.join(BASE_DIR, "outputs", "models")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

WSTS_PATH      = os.path.join(ROOT_DIR, "wsts_historical.xlsx")
MERGED_PATH    = os.path.join(DATA_DIR,  "merged_dataset.csv")
FEATURES_PATH  = os.path.join(DATA_DIR,  "features_dataset.csv")
TUNED_PKL      = os.path.join(MODEL_DIR, "best_xgboost.pkl")
SELECTED_PKL   = os.path.join(MODEL_DIR, "best_xgboost_selected.pkl")
FINAL_PKL      = os.path.join(MODEL_DIR, "best_xgboost_final.pkl")

# FRED API 키 (환경변수에서 로드)
FRED_API_KEY = os.getenv("FRED_API_KEY")

# 데이터 수집 기간
START_DATE = "1993-01-01"
END_DATE   = "2026-03-31"

# 타겟
PRIMARY_TARGET = "TARGET_Worldwide_YoY_T6"

# CV 설정
TEST_EVAL_SIZE = 24   # hold-out 개월
N_SPLITS       = 5    # TimeSeriesSplit fold 수
TEST_SIZE      = 12   # fold 당 test 개월
MIN_TRAIN      = 60   # fold 최소 train 개월
N_TRIALS       = 50   # Optuna trial 수
RANDOM_STATE   = 42

# Asymmetric Loss 가중치 (Bear 오예측에 높은 페널티)
W_BULL_CORRECT = 1.0
W_BULL_WRONG   = 2.0
W_BEAR_CORRECT = 1.5
W_BEAR_WRONG   = 3.0
BEAR_SAMPLE_W  = 2.0   # Bear 월 sample_weight

for d in [DATA_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)
