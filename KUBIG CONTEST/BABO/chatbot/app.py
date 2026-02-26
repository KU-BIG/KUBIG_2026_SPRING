"""
app.py - AI Compliance Assistant Streamlit UI
"""

import streamlit as st
import requests
import re
from datetime import datetime
from typing import List, Dict

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

st.set_page_config(
    page_title="AI Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }

/* ── 채팅 입력창 하단 면책 고지 (입력창 바깥으로 완전 분리) ── */
[data-testid="stChatInput"] {
    position: relative !important;
    margin-bottom: 35px !important; /* 안쪽이 아닌 '바깥쪽' 아래에 35px의 빈 공간을 강제로 만듦 */
}

[data-testid="stChatInput"]::after {
    content: "AI Compliance Checker는 AI이며 법률 정보 제공 시 실수를 할 수 있습니다. 최종 의사결정 전 반드시 법률 전문가와 교차 검증하시기 바랍니다.";
    position: absolute;
    bottom: -28px; /* 마이너스 값을 주어 입력창 테두리 밖으로 텍스트를 완전히 끌어내림 */
    left: 0;
    right: 0;
    text-align: center;
    font-size: 12px;
    color: #999999;
    word-break: keep-all;
    pointer-events: none;
}     

/* ── 기본 테마 및 폰트 ── */
html, body, .stApp {
    font-family: 'Pretendard', -apple-system, sans-serif !important;
}
:root { color-scheme: light !important; }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background-color: #F4F7FD !important;
    color: #1a1f36 !important;
}

/* ── 헤더 ── */
.stDeployButton { display: none !important; }

/* 모든 제목 태그에 붙는 불필요한 링크 없애기 */
a.header-anchor, h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { 
    display: none !important; 
    pointer-events: none !important; 
}

/* ── 사이드바 ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F8;
}
section[data-testid="stSidebar"] .stScrollToBottomContainer,
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
}
[data-testid="stSidebarUserContent"] {
    padding-bottom: 1.5rem !important; 
}

/* ── 2. 사이드바 로고 영역 ── */
.sb-logo {
    display: flex; align-items: center; gap: 14px;
    margin-top: -1.5rem !important; 
    padding-bottom: 1.5rem !important; 
    border-bottom: 1px solid #E2E8F8 !important;
    margin-bottom: 0 !important;
}
.sb-logo-icon {
    width: 52px; height: 52px; 
    flex-shrink: 0; 
    background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; color: white;
    box-shadow: 0 4px 12px rgba(27,79,216,0.2);
}
.sb-logo-text {
    height: 52px; 
    display: flex; flex-direction: column; 
    gap: 2px;
    justify-content: space-between; 
    padding: 4px 0 2px 0; 
}
.sb-logo-name { font-size: 1.1rem; font-weight: 800; color: #1B4FD8; line-height: 1.1; margin: 0; margin-top: -3px !important;} 
.sb-logo-sub  { font-size: 0.75rem; color: #94A3B8; line-height: 1.2; margin: 0;}

/* ── 사이드바 버튼 & 예시질문 ── */
.sidebar-title-wrapper {
    text-align: center; margin-bottom: 1.2rem; margin-top: 0 !important;
}
.sidebar-title-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(191, 219, 254, 0.8);
    box-shadow: 0 4px 12px rgba(27,79,216,0.06);
    backdrop-filter: blur(4px);
    color: #1B4FD8;
    font-size: 1.05rem; font-weight: 800; 
    padding: 8px 20px; 
    border-radius: 24px !important;
}
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 24px !important; 
    background: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid rgba(191, 219, 254, 0.6) !important;
    box-shadow: 0 4px 12px rgba(27,79,216,0.04) !important;
    backdrop-filter: blur(4px) !important;
    font-size: 0.82rem !important; 
    color: #1E293B !important;
    padding: 10px 14px !important;
    text-align: center !important; 
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #FFFFFF !important;
    border-color: #1B4FD8 !important;
    color: #1B4FD8 !important;
    box-shadow: 0 6px 16px rgba(27,79,216,0.12) !important;
    transform: translateY(-2px);
}

/* 스트림릿 버튼 속 숨겨진 글씨(p 태그) 크기 강제 축소 */
section[data-testid="stSidebar"] .stButton > button p {
    font-size: 0.9rem !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}
            
/* ── 메인 컨텐츠 영역 ── */
.block-container {
    padding-top: 3.5rem !important; 
    padding-bottom: 5rem !important; 
    max-width: 900px !important;
}

/* ── 3. 채팅창 메인 헤더 ── */
.app-header {
    display: flex; align-items: center; gap: 18px; 
    padding-top: 10px !important; 
    padding-bottom: 1.5rem !important;
    margin-bottom: 2rem !important;
    border-bottom: 2px solid #E2E8F8 !important;
}
.app-header-icon {
    width: 64px; height: 64px; 
    flex-shrink: 0; 
    background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    border-radius: 16px; 
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; box-shadow: 0 4px 16px rgba(27,79,216,0.28); 
}
.app-header-text {
    height: 64px; 
    display: flex; flex-direction: column; 
    justify-content: space-between; 
    padding: 0 !important; 
}
.app-header h1 {
    font-size: 1.65rem !important; 
    font-weight: 800 !important;
    color: #1B4FD8 !important; letter-spacing: -0.5px;
    line-height: 1.0 !important; 
    margin: 0 !important; 
    margin-top: -18px !important; 
}
.app-header p {
    font-size: 0.95rem !important; 
    color: #64748B; 
    line-height: 1.0 !important;
    margin: 0 !important; 
    position: relative !important;
    top: -5px !important; 
}

/* ── 채팅 래퍼 및 버블 ── */
.chat-wrapper { display: flex; flex-direction: column; gap: 18px; margin-bottom: 20px; }

.msg-user-row {
    display: flex; justify-content: flex-end; align-items: flex-end; gap: 8px;
}
.msg-user-bubble {
    background: linear-gradient(135deg, #1B4FD8, #2563EB);
    color: #FFFFFF; border-radius: 20px 20px 4px 20px;
    padding: 14px 18px; max-width: 70%;
    font-size: 1.0rem; line-height: 1.6;
    box-shadow: 0 3px 12px rgba(27,79,216,0.22); word-break: break-word;
}
.msg-time {
    font-size: 0.8rem; color: #94A3B8; white-space: nowrap; margin-bottom: 4px;
}

.msg-ai-row {
    display: flex; justify-content: flex-start; align-items: flex-start; gap: 10px;
}
.msg-ai-content { max-width: 85%; }
.msg-ai-name {
    font-size: 1.0rem; font-weight: 700;
    color: #1B4FD8; margin-bottom: 6px; letter-spacing: 0.3px;
}
.msg-ai-bubble {
    background: #FFFFFF; color: #1E293B;
    border-radius: 4px 20px 20px 20px;
    padding: 15px 18px;
    font-size: 1.0rem; line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F8; word-break: break-word;
}

/* ── 참고 조항 카드 ── */
.source-label {
    font-size: 0.85rem; font-weight: 700; color: #1B4FD8; margin: 10px 0 8px 0;
}
.source-card {
    background: #F8FAFF; border: 1px solid #DBEAFE;
    border-left: 4px solid #1B4FD8; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px;
}
.source-card-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}
.source-tag {
    background: #DBEAFE; color: #1B4FD8;
    font-size: 0.85rem; font-weight: 700;
    padding: 3px 8px; border-radius: 4px;
}
.source-id { font-weight: 700; color: #1E293B; font-size: 1.0rem; }
.source-title-text { font-size: 0.85rem; color: #64748B; }
.source-excerpt {
    font-size: 0.85rem; color: #475569; line-height: 1.6;
    border-top: 1px solid #E2E8F8; padding-top: 8px; margin-top: 6px;
}

/* ── 제미나이 스타일 둥근 입력창 ── */
[data-testid="stBottom"] {
    background-color: transparent !important;
    padding-bottom: 0 !important; 
}
[data-testid="stBottom"] > div {
    padding-bottom: 1rem !important; 
}
.stChatInput { background-color: transparent !important; }
.stChatInput > div, .stChatInput textarea { background-color: #FFFFFF !important; }
.stChatInput > div {
    border: 2px solid #BFDBFE !important; 
    border-radius: 24px !important; 
    box-shadow: 0 -30px 40px 15px #F4F7FD, 0 2px 12px rgba(27,79,216,0.08) !important;
}
.stChatInput > div:focus-within {
    border-color: #1B4FD8 !important; box-shadow: 0 -30px 40px 15px #F4F7FD, 0 2px 18px rgba(27,79,216,0.15) !important;
}
button[data-testid="stChatInputSubmit"] svg {
    stroke: #1B4FD8 !important; fill: #1B4FD8 !important; color: #1B4FD8 !important;
}
button[data-testid="stChatInputSubmit"]:hover, 
button[data-testid="stChatInputSubmit"]:focus,
button[data-testid="stChatInputSubmit"]:active {
    background-color: #EFF6FF !important; border-color: transparent !important;
}

/* ── 기타 ── */
details { background: transparent !important; border: none !important; }
details summary {
    font-size: 0.95rem !important; font-weight: 600 !important;
    color: #1B4FD8 !important; padding: 6px 0 !important;
}

.empty-state { text-align: center; padding: 60px 20px 0px 20px; color: #94A3B8; }
.empty-state-icon { font-size: 3rem; margin-bottom: 14px; }
.empty-state h3 { font-size: 1.2rem; font-weight: 700; color: #1E293B; margin-bottom: 8px; }
.empty-state p { font-size: 0.92rem; line-height: 1.65; }
hr { border-color: #E2E8F8 !important; margin: 1.5rem 0 !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
            
/* 1. 하단 푸터(Made with Streamlit) 공간 완전 삭제 */
footer {
    display: none !important; 
}

/* 2. 채팅 입력창을 바닥으로 끌어내리기 */
div[data-testid="stChatInput"] {
    padding-bottom: 0px !important;
    margin-bottom: -30px !important;
}
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

def build_history_payload() -> List[Dict]:
    history = []
    for msg in st.session_state.messages[:-1]:
        role = msg["role"]
        content = msg["content"]
        if role in ("user", "assistant"):
            history.append({"role": role, "content": content})
    return history

def call_chat(question: str) -> Dict:
    history = build_history_payload()
    try:
        r = requests.post(
            CHAT_ENDPOINT,
            json={"question": question, "history": history},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"오류 발생: {str(e)}"}

def now_str() -> str:
    return datetime.now().strftime("%H:%M")

def render_sources_html(sources: List[Dict]) -> str:
    if not sources:
        return ""
    html = '<div class="source-label">참고 조항 상세</div>'
    for src in sources:
        raw_id = src.get("source_id", "")
        if "::" in raw_id:
            law_name, article = raw_id.split("::", 1)
        else:
            law_name, article = "관련 법안", raw_id

        title = f'<span class="source-title-text"> — {src.get("title","")}</span>' if src.get("title") else ""
        
        excerpt = src.get("excerpt", "")
        exc_html = f'<div class="source-excerpt">"{excerpt}"</div>' if excerpt else ""
        
        html += f"""
        <div class="source-card">
            <div class="source-card-header">
                <span class="source-tag">{law_name}</span>
                <span class="source-id">{article}</span>{title}
            </div>{exc_html}
        </div>"""
    return html

def is_source_referenced(raw_id: str, content: str) -> bool:
    """조항 번호뿐만 아니라 '어느 법안'인지 문맥을 역추적하여 교차 검증하는 스마트 함수"""
    content_lower = content.lower()
    raw_id_lower = raw_id.lower()
    
    # 원본 ID에서 법안명과 조항 번호 분리 (예: "Korea AI Law", "Article 2")
    law_name = raw_id.split("::")[0].strip().lower() if "::" in raw_id else ""
    article_num = raw_id.split("::")[-1].strip().lower() if "::" in raw_id else raw_id_lower
    
    is_eu = "eu" in law_name or "europe" in law_name or "유럽" in law_name
    is_kr = "korea" in law_name or "한국" in law_name or "기본법" in law_name
    
    # 1. 완전 일치 방어 (예: "Korea AI Law::Article 2" 자체가 텍스트에 있을 때)
    if raw_id_lower in content_lower:
        return True
        
    patterns_to_check = []
    
    # 정규식 패턴 생성
    art_match = re.search(r'article\s+(\d+)', article_num)
    if art_match:
        num = art_match.group(1)
        patterns_to_check.extend([rf"제\s*{num}\s*조", rf"article\s*{num}"])
        
    annex_match = re.search(r'annex\s+([ivx]+|\d+)', article_num)
    if annex_match:
        val = annex_match.group(1)
        roman_to_arabic = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'}
        arabic = roman_to_arabic.get(val, val)
        patterns_to_check.extend([rf"부속서\s*{arabic}", rf"annex\s*{arabic}", rf"부속서\s*{val}", rf"annex\s*{val}"])
        
    recital_match = re.search(r'recital\s+(\d+)', article_num)
    if recital_match:
        num = recital_match.group(1)
        patterns_to_check.extend([rf"recital\s*{num}", rf"리사이틀\s*{num}"])
        
    if not patterns_to_check:
        return article_num in content_lower
        
    # 2. 본문에서 해당 번호(예: 제2조)가 등장할 때마다 어느 법안인지 교차 검증!
    for pattern in patterns_to_check:
        for match in re.finditer(pattern, content_lower):
            idx = match.start()
            
            # 조항 번호 등장 위치에서 역방향으로 가장 가까운 EU/한국 키워드 탐색
            eu_idx = max(content_lower.rfind("eu", 0, idx), content_lower.rfind("유럽", 0, idx))
            kr_idx = max(content_lower.rfind("한국", 0, idx), content_lower.rfind("기본법", 0, idx))
            
            # 이 "제2조"가 과연 EU법인가 한국법인가?
            if is_kr:
                if kr_idx >= eu_idx or kr_idx != -1: # 한국법 키워드가 더 가깝거나 존재하면 합격
                    if kr_idx >= eu_idx:
                        return True
            elif is_eu:
                if eu_idx >= kr_idx or eu_idx != -1: # EU법 키워드가 더 가깝거나 존재하면 합격
                    if eu_idx >= kr_idx:
                        return True
            else:
                return True # 법안 식별이 불가능한 예외 케이스
                
    return False

def render_message(msg: Dict):
    role = msg["role"]
    content = msg["content"]
    t = msg.get("time", "")

    if role == "user":
        st.markdown(f"""
        <div class="msg-user-row">
            <span class="msg-time">{t}</span>
            <div class="msg-user-bubble">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        sources = msg.get("sources", [])
        
        display_content = content.replace("\n", "<br>")
        if "- 규제 대상:" in display_content:
            display_content = re.sub(
                r'(<br>\s*)+-\s*규제 대상:', 
                "<hr style='margin: 0px 0 1.2rem 0; border: none; border-top: 1px solid #E2E8F8;'>- 규제 대상:", 
                display_content
            )

        st.markdown(f"""
        <div class="msg-ai-row">
            <div class="msg-ai-content">
                <div class="msg-ai-name">AI Compliance Checker</div>
                <div class="msg-ai-bubble">{display_content}</div>
                <div class="msg-time" style="margin-top:5px;">{t}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if sources:
            grouped_sources = {}
            
            for src in sources:
                raw_id = src.get("source_id", "")
                excerpt = src.get("excerpt", "")
                
                if len(excerpt.strip()) < 15 or excerpt.strip().lower() in raw_id.strip().lower():
                    continue
                
                # 병합 키(Key)를 조항 번호가 아닌 '법안명+조항번호' 전체로 설정해 덮어쓰기 방지
                group_key = raw_id.strip().upper()
                
                # 이제 is_source_referenced 함수에 raw_id 전체를 넘겨줍니다.
                if is_source_referenced(raw_id, content):
                    if group_key not in grouped_sources:
                        grouped_sources[group_key] = src.copy()
                    else:
                        existing_excerpt = grouped_sources[group_key]["excerpt"]
                        if excerpt not in existing_excerpt and existing_excerpt not in excerpt: 
                            grouped_sources[group_key]["excerpt"] = existing_excerpt + "<br><br>[이어진 내용] " + excerpt
            
            filtered_sources = list(grouped_sources.values())
            
            if filtered_sources:
                label = f"참고 조항 {len(filtered_sources)}건"
                with st.expander(label):
                    st.markdown(render_sources_html(filtered_sources), unsafe_allow_html=True)

# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🪄</div>
        <div class="sb-logo-text">
            <div class="sb-logo-name">AI Compliance Checker</div>
            <div class="sb-logo-sub">AI 비즈니스 및 서비스 도입을 위한<br>
                규제 검토 어시스턴트</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-title-wrapper">
        <span class="sidebar-title-badge">추천 질문</span>
    </div>
    """, unsafe_allow_html=True)

    examples = [
        "인사 평가 시스템에 AI를 연동하면 고위험 AI에 해당하나요?",
        "유럽 시장에 안면 인식 AI 서비스를 출시할 때 주의할 점은?",
        "국내 고객센터에 생성형 AI를 도입할 때 지켜야 할 인공지능기본법상 의무는?",
        "회사가 보유한 데이터를 사용자의 명시적인 동의 없이 학습에 이용할 경우 페널티나 과징금은?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:14]}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("대화 초기화", use_container_width=True, key="clear"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────
# 메인 헤더 & 입력창 처리
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🪄</div>
    <div class="app-header-text">
        <h1>AI COMPLIANCE CHECKER</h1>
        <p>EU AI Act · 한국 인공지능기본법 기반 AI 시스템 규제 진단 및 의무 사항 안내</p>
    </div>
</div>
""", unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", None)
placeholder_text = prefill if prefill else "도입하려는 AI 시스템의 기능, 목적, 타겟 국가 등을 자세히 입력해 보세요..."
user_input = st.chat_input(placeholder_text)

if prefill and not user_input:
    user_input = prefill

if user_input and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "time": now_str(),
    })

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⚖️</div>
        <h3>어떤 AI 시스템 도입을 검토 중이신가요?</h3>
        <p>신규 AI 서비스를 기획하거나 기존 시스템에 AI 연동을 준비 중이시라면, 적용하려는 국가와 기술(목적)을 알려주세요.<br>
        EU AI Act 및 한국 인공지능기본법을 바탕으로 규제 등급, 핵심 의무, 위반 리스크를 명확한 법적 근거와 함께 진단해 드립니다.</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)

if user_input and user_input.strip():
    with st.spinner("관련 법령을 바탕으로 규제 리스크를 분석하고 있습니다..."):
        result = call_chat(question)

    if "error" in result:
        answer = result["error"]
        sources = []
    else:
        answer = result.get("answer", "답변을 생성하지 못했습니다.")
        answer = answer.replace("[Context]", "").replace("제공된 데이터", "관련 법안")
        sources = result.get("sources", [])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "time": now_str(),
    })
    
    st.rerun()
