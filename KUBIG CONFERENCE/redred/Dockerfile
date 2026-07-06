# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
# Streamlit 대시보드용 이미지
# 모델 산출물(pkl/csv)은 이미지에 포함하지 않고 런타임에 S3에서 받음
# ─────────────────────────────────────────────────────────────

# 가볍고 호환성 좋은 공식 슬림 이미지 사용
FROM python:3.11-slim

# xgboost 등 OpenMP 의존 라이브러리 구동에 필요한 libgomp1 설치
# slim 이미지에는 빠져 있어 미설치 시 import 단계에서 OSError 발생
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 레이어 캐싱 최적화 ──
# 의존성 목록만 먼저 복사·설치 → 소스만 바뀌면 pip 설치 캐시 재사용
# stage1/requirements.txt를 stage1·stage2 공통 의존성으로 사용
COPY stage1/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── 소스 복사 ──
# 의존성 설치 이후 단계라 소스 변경이 위 캐시를 무효화하지 않음
COPY stage1/ ./stage1/
COPY stage2/ ./stage2/
COPY app.py ./app.py

# WSTS 원본 데이터는 이미지에 포함 (S3 다운로드 대상 아님)
COPY wsts_historical.xlsx ./wsts_historical.xlsx

# Streamlit 기본 포트
EXPOSE 8501

# 0.0.0.0 바인딩으로 컨테이너 외부에서 접근 가능하게 실행
# Railway 등 PaaS가 주입하는 $PORT를 우선 사용, 없으면 8501로 폴백
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

