"""
main.py - FastAPI 백엔드 서버
AI Compliance RAG 시스템의 API 엔드포인트를 제공합니다.
"""

import sys
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

rag_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    print("🚀 서버 시작: AI Compliance RAG 엔진을 초기화합니다...")
    try:
        from src.rag_engine import AIComplianceRAG
        rag_instance = AIComplianceRAG()
        print("✅ RAG 엔진 초기화 완료!")
    except Exception as e:
        print(f"❌ RAG 엔진 초기화 실패: {e}")
    yield
    print("🛑 서버 종료.")

app = FastAPI(
    title="AI Compliance Assistant API",
    description="EU AI Act 기반 AI 컴플라이언스 분석 및 답변 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HistoryMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[HistoryMessage]] = []

class SourceReference(BaseModel):
    source_type: str
    source_id: str
    title: Optional[str] = ""
    excerpt: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    sub_queries: Optional[List[str]] = []

def parse_sources(raw_context: str) -> List[SourceReference]:
    """
    [핵심 수정 내용]
    첫 줄만 읽고 break 하던 버그를 고치고,
    re.DOTALL을 사용해 내용(Chunk)의 '모든 줄'을 끝까지 긁어옵니다.
    """
    sources: List[SourceReference] = []
    blocks = raw_context.split("--- 출처:")
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # 첫 줄(헤더)과 나머지(본문) 분리
        parts = block.split("\n", 1)
        header_line = parts[0].replace("---", "").strip()
        body = parts[1] if len(parts) > 1 else ""

        # 정규식(DOTALL)으로 '내용(Chunk):' 부터 다음 섹션 전까지 모든 줄 추출
        chunk_match = re.search(r"내용\(Chunk\):\s*(.*?)(?:\n관련 구조\(Graph\):|$)", body, re.DOTALL)
        chunk_text = chunk_match.group(1).strip() if chunk_match else ""

        match = re.match(r"\[(.*?)\]\s+([^\(]+)(?:\((.*?)\))?", header_line)
        if match:
            src_type = match.group(1).strip()
            src_id = match.group(2).strip()
            src_title = match.group(3).strip() if match.group(3) else ""
        else:
            src_type = "참조"
            src_id = header_line[:50]
            src_title = ""

        # 추출한 텍스트가 너무 길면 자르기
        excerpt_text = chunk_text[:500] + ("..." if len(chunk_text) > 500 else "")

        is_duplicate = False
        for s in sources:
            if s.source_id == src_id and s.excerpt == excerpt_text:
                is_duplicate = True
                break
                
        if not is_duplicate and chunk_text:
            sources.append(SourceReference(
                source_type=src_type,
                source_id=src_id,
                title=src_title,
                excerpt=excerpt_text
            ))
            
    return sources

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_engine_ready": rag_instance is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG 엔진이 초기화되지 않았습니다.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    try:
        history_dicts = [{"role": h.role, "content": h.content} for h in request.history]
        answer, raw_context = rag_instance.generate_answer(question, history=history_dicts)
        sources = parse_sources(raw_context)
        sub_queries = rag_instance.analyze_and_route_query(question)

        return ChatResponse(
            answer=answer,
            sources=sources,
            sub_queries=sub_queries,
        )

    except Exception as e:
        print(f"❌ /chat 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")