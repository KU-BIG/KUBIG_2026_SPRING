import sys
import os

# 경로 추가 코드: 현재 폴더와 부모 폴더를 모두 파이썬 길찾기에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate

# 1. 임포트
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 2. 글자 수(max_tokens)를 늘리기 위해 랭체인 모듈만 추가로 가져옵니다.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 프로젝트 경로 추가 및 RAG 엔진 불러오기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rag_engine import AIComplianceRAG

def run_evaluation():
    print("🧪 AI Compliance RAG 평가 파이프라인 가동 중...")
    rag_engine = AIComplianceRAG()

    # 3. 심판관 커스텀
    # 평가 도중 글자가 잘리지 않게 max_tokens를 넉넉하게 8192로 늘립니다.
    # 임베딩 에러 방지를 위해 명시적으로 임베딩 모델도 쥐여줍니다.
    print("⚙️ 심판관 LLM의 글자 수 제한을 해제합니다...")
    my_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=8192, temperature=0.0)
    my_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. 10개 QA 데이터셋 불러오기
    qa_path = os.path.join('src', 'qa_dataset.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # [중요] 최초 테스트용으로 3개만 먼저 실행해 봅니다.
    # qa_data = qa_data[:3] 

    print(f"🚀 총 {len(qa_data)}개의 테스트 케이스에 대해 답변 생성을 시작합니다...")

    # RAGAS가 요구하는 데이터 형식(Dictionary of Lists) 준비
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # 2. 우리 RAG 엔진으로 답변 추출
    for item in tqdm(qa_data, desc="답변 생성 중"):
        question = item['question']
        ground_truth = item['answer']
        
        answer, context_text = rag_engine.generate_answer(question)
        data_samples["question"].append(question)
        data_samples["answer"].append(answer)
        data_samples["contexts"].append([context_text]) 
        data_samples["ground_truth"].append(ground_truth)

    # 3. 데이터셋 변환 및 평가 실행
    print("\n⚖️ RAGAS 심판관 모델이 지표를 채점하고 있습니다 (잠시만 기다려주세요)...")
    dataset = Dataset.from_dict(data_samples)

    # 4. 평가 수행: 원본 구조를 유지하되, 방금 만든 my_llm만 옵션으로 투입
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=my_llm,                 # 글자 수 넉넉한 LLM 투입
        embeddings=my_embeddings,   # 에러 안 나는 임베딩 투입
        raise_exceptions=False      # 중간 에러 무시
    )

    # 4. 결과 출력 및 엑셀(CSV) 저장
    print("\n📊 [최종 평가 평균 점수]")
    print(score)

    df_score = score.to_pandas()
    result_filename = "rag_evaluation_results.csv"
    df_score.to_csv(result_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 문항별 평가 결과가 '{result_filename}' 파일로 저장되었습니다!")

if __name__ == "__main__":
    run_evaluation()