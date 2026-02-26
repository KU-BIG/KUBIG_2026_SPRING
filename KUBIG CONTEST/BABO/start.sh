#!/bin/bash
cd /workspaces/aicompliance

echo "📦 패키지 설치 중..."
pip install -r requirements.txt -q
pip install uvicorn streamlit -q

echo "🚀 FastAPI 서버 시작..."
cd /workspaces/aicompliance/chatbot
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 &

echo "⏳ FastAPI 준비 완료까지 대기 중..."
MAX_WAIT=300   # 최대 5분 대기
ELAPSED=0
until curl -s http://localhost:8000/health | grep -q '"status":"ok"'; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "❌ FastAPI 서버가 ${MAX_WAIT}초 내에 응답하지 않았습니다. 로그를 확인하세요."
        exit 1
    fi
    echo "   ... 대기 중 (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "✅ FastAPI 준비 완료! (${ELAPSED}s 소요)"

echo "🎨 Streamlit 시작..."
cd /workspaces/aicompliance
python -m streamlit run chatbot/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
