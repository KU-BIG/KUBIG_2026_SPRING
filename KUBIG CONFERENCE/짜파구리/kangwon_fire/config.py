from pathlib import Path

BASE_DIR  = Path(__file__).parent
ROOT_DIR  = BASE_DIR.parent
DATA_DIR  = ROOT_DIR / "강원도"

AWS_DIR       = DATA_DIR / "AWS"
ASOS_DIR      = DATA_DIR / "ASOS"
LIGHTNING_DIR = DATA_DIR / "낙뢰"
AWS_META      = DATA_DIR / "AWS_META_STATION.csv"
ASOS_META     = DATA_DIR / "ASOS_META_STATION.csv"

POLES_CSV    = ROOT_DIR / "전주데이터_DEM경사향추가.csv"
TERRAIN_CSV  = ROOT_DIR / "전주데이터_terrain_score.csv"
WILDFIRE_CSV = ROOT_DIR / "화재종별_신고재난_2022.csv"

OUT_DIR = ROOT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

FIRE_SEASON_MONTHS = [11, 12, 1, 2, 3, 4, 5]

ASOS_COLS = [
    "지점", "일시",
    "최소 상대습도(%)", "평균 상대습도(%)",
    "합계 일조시간(hr)", "합계 일사량(MJ/m2)",
    "평균 해면기압(hPa)", "평균 지면온도(°C)",
    "합계 대형증발량(mm)", "일강수량(mm)",
    "일 최심신적설(cm)",
]

FEATURES = [
    "FFMC", "ISI", "BUI", "FWI",
    "dryness_index", "drought_days", "dry_wind_streak",
    "최고기온(°C)", "temp_range",
    "최대 순간 풍속(m/s)", "평균 풍속(m/s)", "wind_ma3",
    "wind_sin", "wind_cos",
    "평균 해면기압(hPa)",
    "lightning_count_2km", "lightning_max_intensity_2km", "lightning_dry_compound",
    "month_sin", "month_cos",
]
