# KUBIG AI Compliance Checker
**GraphRAG 기반 AI 규제 검토 챗봇**

한국 인공지능기본법(2026.1.)과 EU AI Act(2024.8.) 발효에 따라, AI 시스템을 도입하려는 기업이 규제 등급·핵심 의무·위반 리스크를 즉시 진단받을 수 있는 법률 컨설팅 챗봇입니다. 규제 조항 간 참조 관계를 Neo4j 지식 그래프로 구조화하고, Hybrid GraphRAG 파이프라인을 통해 문서 근거가 명확한 답변을 제공합니다.

---

## Project Structure

```
aicompliance/
├── chatbot/                       # 🚀 서비스 메인 (실행 대상)
│   ├── app.py                     # Streamlit 프론트엔드 UI
│   ├── main.py                    # FastAPI 백엔드 서버
│   ├── .streamlit/
│   │   └── config.toml            # Streamlit 설정
│   └── src/
│       ├── db_connection.py       # Neo4j · OpenAI 연결 설정
│       ├── rag_engine.py          # RAG 엔진 (검색 · 리랭킹 · 답변 생성)
│       ├── qa_dataset.json        # Few-shot QA 데이터셋 (고위험 AI 시나리오 10개)
│       ├── batch_test_rag.py      # 스트레스 테스트 스크립트 (OOD · 모호한 질의 등)
│       └── evaluate_rag.py        # Ragas 기반 성능 평가 스크립트
│
├── src/                           # 📦 데이터 전처리 파이프라인
│   ├── ingest_EU_legal.py         # EU AI Act HTML 파싱 및 청킹
│   ├── ingest_KR_legal.py         # 한국 AI 기본법 PDF 파싱 및 청킹
│   ├── extract_relations_aiact.py # EU AI Act 노드/릴레이션 추출
│   ├── extract_relations_aicorelaw.py  # 한국 AI 기본법 노드/릴레이션 추출
│   ├── merge_relations.py         # 관계 데이터 병합
│   ├── add_embeddings.py          # 벡터 임베딩 생성 (text-embedding-3-small)
│   ├── add_embeddings_from_save.py
│   ├── upload_to_neo4j.py         # Neo4j AuraDB 업로드
│   ├── audit_graph_data.py        # 그래프 데이터 검증
│   └── prepare_finetune_data.py   # 파인튜닝 데이터 준비
│
├── corpus.jsonl                   # 전처리된 법률 코퍼스
├── finetune_qwen_experiment.py    # Qwen 0.5B LoRA 파인튜닝 실험
├── FORMAT_SPEC.md                 # 데이터 포맷 명세
├── requirements.txt               # Python 의존성
├── devcontainer.json              # GitHub Codespaces 설정
├── start.sh                       # 서버 실행 스크립트
└── README.md
```

> 서비스 실행에 필요한 파일은 모두 `chatbot/` 폴더에 있습니다. `src/`는 데이터 전처리용입니다.

---

## Architecture

```
사용자 질의
    ↓
[Streamlit UI] app.py
    ↓  HTTP POST /chat
[FastAPI] main.py
    ↓
[RAG Engine] rag_engine.py
    │
    ├── 1단계: 질의 라우팅 (Query Routing)
    │         복잡한 질문을 2~3개의 Sub-query로 분할
    │         '벌금/제재' 키워드 감지 시 전용 쿼리 강제 생성
    │
    ├── 2단계: 도메인 키워드 변환 (Query Translation)
    │         한국어 질의 → 영문 법률 용어로 변환 및 확장
    │         (예: '인사/채용' → 'Employment, High-risk AI')
    │
    ├── 3단계: 하이브리드 탐색 (Hybrid Retrieval)
    │         Vector Search: 코사인 유사도 기반 1차 탐색 (k=50)
    │         Graph Traversal: 연관 처벌·의무 조항 연쇄 추출
    │
    ├── 4단계: 리랭킹 & 조항 주입 (Reranking & Injection)
    │         Cross-Encoder(ko-reranker)로 상위 15개 정밀 선별
    │
    └── 5단계: 프롬프트 조립 & 답변 생성 (Prompt Assembly)
              Dynamic Few-shot + 대화 맥락 + 법안 원문 결합
              구조화된 컨설팅 리포트 출력
                  (결론 → 법적 근거 → 역질문 → 3줄 요약)
```

---

## Quick Start

### 1. 환경변수 설정

Codespaces 터미널에서 `.env` 파일을 생성합니다.

```bash
cat > /workspaces/aicompliance/.env << 'EOF'
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
EOF
```

> `.env` 파일은 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다.

### 2. 서버 실행

```bash
bash start.sh
```

`start.sh` 실행 순서:
1. 패키지 설치 (`requirements.txt`)
2. FastAPI 백엔드 서버 시작 (port 8000)
3. `/health` 엔드포인트로 서버 준비 완료 확인 (최대 5분 대기)
4. Streamlit UI 시작 (port 8501)

### 3. 접속

| 서비스 | 주소 |
|--------|------|
| Streamlit UI | `http://localhost:8501` |
| FastAPI 서버 | `http://localhost:8000` |
| API 문서 | `http://localhost:8000/docs` |

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| RAG Framework | LangChain |
| GraphDB | Neo4j AuraDB |
| Embedding | OpenAI text-embedding-3-small |
| Reranking | Dongjin-kr/ko-reranker (Cross-Encoder) |
| LLM | OpenAI GPT (API) |

---

## Performance (Ragas)

| 지표 | 평균 점수 |
|------|----------|
| Faithfulness | 0.7966 |
| Answer Relevancy | 0.1067 |
| Context Precision | 1.0000 |
| Context Recall | 0.5910 |

---

## Notes

- `.env` 파일은 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다.
- 최종 답변 모델은 파인튜닝 없이 사전학습된 LLM API를 사용합니다. (Qwen 0.5B 파인튜닝 실험 결과 mean_token_accuracy 0.42로 성능 한계 확인)
- 본 챗봇은 법률 정보 제공을 목적으로 하며, 최종 의사결정 전 반드시 법 전문가와 교차 검증하시기 바랍니다.

