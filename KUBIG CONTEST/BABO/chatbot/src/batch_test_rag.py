import os
import sys
import pandas as pd
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import AIComplianceRAG

def run_stress_test():
    print("⚙️ AI Compliance RAG 스트레스 테스트 가동 중...")
    rag = AIComplianceRAG()

    # 3가지 난이도별 테스트 질문 세트
    test_queries = [
        # 🟢 [좋은 질문] 구체적이고 명확한 질문 (정확한 법리 해석 테스트)
        "의료 기기로 사용되는 AI 진단 보조 시스템을 유럽 시장에 출시하려 합니다. 환자의 민감한 생체 데이터를 처리하는데, 이거 고위험 AI에 해당하나요?",
        "한국에서 딥페이크 탐지 기술을 개발 중입니다. 인공지능 기본법상 고영향 AI에 해당하는지, 투명성 의무는 어떻게 지켜야 하는지 궁금합니다.",
        
        # 🟡 [그저 그런 질문] 너무 포괄적이거나 정보가 부족한 질문 (역질문 및 유도 능력 테스트)
        "우리 회사에서 이번에 AI 도입할 건데 불법인가요?",
        "유럽에서 AI 쓰다가 걸리면 벌금 얼마 내야 돼요?",
        "채용할 때 쓰는 AI 시스템 규제 좀 알려주세요.",

        # 🔴 [나쁜 질문] 엉뚱하거나 제공된 법안 범위를 벗어난 질문 (할루시네이션 차단 및 방어력 테스트)
        "미국 캘리포니아주 AI 규제법(SB 1047)에 따르면 우리 회사가 내야 할 벌금은 얼마인가요?",
        "사내 식당 오늘 점심 메뉴를 추천해주는 AI를 도입했는데, 이것도 인간에게 영향을 미치니까 3500만 유로 과징금 맞나요?",
        "이재용 회장이 AI 시스템 도입할 때 유럽 법안을 어떻게 생각할까요?"
    ]

    results = []
    
    print(f"\n🚀 총 {len(test_queries)}개의 다이나믹 퀘스트에 대한 답변 생성을 시작합니다...\n")

    for i, q in enumerate(tqdm(test_queries, desc="답변 생성 중")):
        try:
            # 1. 어떤 유형의 질문인지 판별 (질문 순서 기반)
            if i < 2:
                q_type = "🟢 좋은 질문"
            elif i < 5:
                q_type = "🟡 그저그런 질문"
            else:
                q_type = "🔴 나쁜 질문"

            # 2. 엔진을 돌려 답변 추출
            answer, context = rag.generate_answer(q)
            
            # 3. 결과 저장
            results.append({
                "유형": q_type,
                "질문": q,
                "생성된 답변": answer,
                "검색된 Context": context
            })
        except Exception as e:
            print(f"\n❌ 에러 발생 (질문: {q}) -> {e}")

    # 4. CSV로 저장
    df = pd.DataFrame(results)
    output_filename = "rag_stress_test_results.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 스트레스 테스트 완료! 결과가 '{output_filename}' 파일로 저장되었습니다.")
    print("엑셀로 열어서 '나쁜 질문'에 챗봇이 어떻게 철벽을 치는지 확인해 보세요!")

if __name__ == '__main__':
    run_stress_test()