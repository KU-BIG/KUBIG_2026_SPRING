"""공통 경로·상수. 모든 sk* 스크립트에서 import해 사용."""
import os, sys

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # skhynix/
ROOT_DIR      = os.path.dirname(BASE_DIR)                     # 프로젝트 루트
MODEL_DIR_REF = os.path.join(ROOT_DIR, "model")
sys.path.insert(0, MODEL_DIR_REF)

# ── Stage 1 참조 (복사 금지) ───────────────────────────────────
FEATURES_PATH    = os.path.join(MODEL_DIR_REF, "outputs", "data",   "features_dataset.csv")
SUPPLY_FINAL_PKL = os.path.join(MODEL_DIR_REF, "outputs", "models", "best_xgboost_final.pkl")

# ── skhynix/ 출력 경로 ────────────────────────────────────────
SK_DATA_DIR   = os.path.join(BASE_DIR, "outputs", "data")
SK_FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
SK_METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

OOS_PRED_PATH = os.path.join(SK_DATA_DIR, "wsts_oos_preds.parquet")
PRICE_PATH    = os.path.join(SK_DATA_DIR, "hynix_price.parquet")
STAGE2_PATH   = os.path.join(SK_DATA_DIR, "stage2_features.parquet")

# ── 타겟 ─────────────────────────────────────────────────────
TARGET_COL  = "hynix_fwd6"           # 6개월 forward return
TARGET_NAME = "6-Month Forward Return"

# ── CV 설정 (월별) ────────────────────────────────────────────
MIN_TRAIN_M  = 60    # 월별 walk-forward 최소 학습 기간
STEP_M       = 1     # walk-forward step (월)
RANDOM_STATE = 42

# ── 거시경제 피처 후보 (features_dataset.csv 실제 컬럼명 기준) ─
# 미존재 컬럼은 sk2_features.py에서 경고 후 자동 스킵
MACRO_FEATURE_CANDIDATES = [
    "FedFunds_lag6",           # DFF_lag6 대응
    "FRED_T10Y2Y_lag6",        # T10Y2Y_lag6 대응
    "FRED_ConsSenti_YoY_lag6", # UMCSENT_lag6 대응
    "FedFunds_diff12",         # DFF_chg_12m 대응 (lag 없는 변형)
    "FRED_T10Y2Y_chg3",        # T10Y2Y_chg_6m 대응 (chg3만 존재)
    "Worldwide_YoY_ma12",      # ww_yoy_roll12_mean 대응
    # 미존재 (자동 스킵): VIXCLS_lag6, BAA10Y_lag6, M2_yoy_lag6, CAPUTLG3344S_lag6
]

for d in [SK_DATA_DIR, SK_FIG_DIR, SK_METRIC_DIR]:
    os.makedirs(d, exist_ok=True)
