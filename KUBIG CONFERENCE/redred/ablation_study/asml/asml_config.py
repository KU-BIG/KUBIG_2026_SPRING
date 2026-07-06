"""공통 경로·상수. 모든 asml* 스크립트에서 import해 사용."""
import os, sys

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # asml/
ROOT_DIR      = os.path.dirname(BASE_DIR)                     # 프로젝트 루트
MODEL_DIR_REF = os.path.join(ROOT_DIR, "model")
SKHYNIX_DIR   = os.path.join(ROOT_DIR, "skhynix")
sys.path.insert(0, MODEL_DIR_REF)

# ── Stage 1 참조 (복사 금지) ──────────────────────────────────
FEATURES_PATH    = os.path.join(MODEL_DIR_REF, "outputs", "data",   "features_dataset.csv")
SUPPLY_FINAL_PKL = os.path.join(MODEL_DIR_REF, "outputs", "models", "best_xgboost_final.pkl")

# ── skhynix/ 참조 (복사 금지) ─────────────────────────────────
OOS_PRED_PATH    = os.path.join(SKHYNIX_DIR, "outputs", "data",    "wsts_oos_preds.parquet")
SK5_RESULTS_PATH = os.path.join(SKHYNIX_DIR, "outputs", "metrics", "sk_crossfirm_results.csv")

# ── asml/ 출력 경로 ───────────────────────────────────────────
ASML_DATA_DIR      = os.path.join(BASE_DIR, "outputs", "data")
ASML_FIG_DIR       = os.path.join(BASE_DIR, "outputs", "figures")
ASML_METRIC_DIR    = os.path.join(BASE_DIR, "outputs", "metrics")
ASML_FEATURES_PATH = os.path.join(ASML_DATA_DIR, "asml_features.parquet")

# ── 타겟·공급 신호 ────────────────────────────────────────────
TARGET_COL  = "ASML_fwd6"
SUPPLY_COL  = "wsts_pred_t6"
FIRM_TICKER = "ASML"

# ── CV 설정 ───────────────────────────────────────────────────
MIN_TRAIN_M  = 60
RANDOM_STATE = 42

# ── FRED API ──────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── 피처 후보 (features_dataset.csv 실제 컬럼명 기준) ─────────
MACRO_COLS = [
    "FedFunds_lag6",
    "FRED_T10Y2Y_lag6",
    "FRED_ConsSenti_YoY_lag6",
    "FedFunds_diff12",
    "FRED_T10Y2Y_chg3",
    "Worldwide_YoY_ma12",
]
SOX_COLS = [
    "Ret_SOX_lag6",
    "Ret_SOX_ma6",
    "Ret_SOX_vol6",
]

for d in [ASML_DATA_DIR, ASML_FIG_DIR, ASML_METRIC_DIR]:
    os.makedirs(d, exist_ok=True)
