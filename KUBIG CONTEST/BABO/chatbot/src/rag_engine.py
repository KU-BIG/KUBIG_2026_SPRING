import sys
import os

# 맥북(macOS) 딥러닝 라이브러리 데드락 방지 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["USE_TF"] = "0"      
os.environ["USE_TORCH"] = "1"   

import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.db_connection import GraphDBConnection
from sentence_transformers import CrossEncoder

class AIComplianceRAG:
    def __init__(self):
        db_conn = GraphDBConnection()
        self.graph = db_conn.get_graph()
        self.embeddings = db_conn.get_embeddings() 
        self.llm = db_conn.get_llm()
        
        print("⏳ Cross-Encoder 리랭킹 모델 로딩 중...")
        self.reranker = CrossEncoder('Dongjin-kr/ko-reranker') 
        self.qa_dataset = self.load_qa_dataset()

    def load_qa_dataset(self) -> List[dict]:
        file_path = os.path.join(os.path.dirname(__file__), 'qa_dataset.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ QA 데이터셋을 불러오지 못했습니다. 파일 위치를 확인하세요: {e}")
            return []

    def get_few_shot_examples(self, query: str, k: int = 2) -> str:
        if not self.qa_dataset:
            return ""
        
        query_vector = self.embeddings.embed_query(query)
        scored_examples = []
        
        for qa in self.qa_dataset:
            if qa['question'].strip() == query.strip():
                continue
                
            qa_vector = self.embeddings.embed_query(qa['question'])
            similarity = np.dot(query_vector, qa_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(qa_vector))
            scored_examples.append((similarity, qa))
            
        top_examples = sorted(scored_examples, key=lambda x: x[0], reverse=True)[:k]
        
        few_shot_text = ""
        for i, (_, qa) in enumerate(top_examples):
            few_shot_text += f"\n[참고 예시 {i+1}]\nQ: {qa['question']}\nA: {qa['answer']}\n"
        
        return few_shot_text

    def analyze_and_route_query(self, query: str) -> List[str]:
        router_prompt = ChatPromptTemplate.from_template("""
        당신은 기업의 AI 컴플라이언스 분석가입니다. 사용자의 질문을 분석하여 데이터베이스 검색에 최적화된 하위 질문(Sub-queries) 리스트로 변환하십시오.
        
        [분할 핵심 지침]
        - 사용자의 질문에 "벌칙", "벌금", "제재", "위반 시", "페널티", "리스크" 등의 단어가 포함되어 있다면, 반드시 그 제재 내용만을 묻는 명확한 하위 질문 1개를 리스트에 추가해야 합니다.
        
        [분류 및 분할 기준]
        1. 단순 포괄형: 핵심 키워드만 뽑아 1개의 질문으로 정리.
        2. 단일 심층형: 특정 조항이나 벌금을 깊게 파고들 수 있도록 1개의 구체적 질문으로 변환.
        3. 다중 분할형: 규제 대상 여부, 핵심 의무, 벌칙 등을 복합적으로 묻는 긴 질문은 2~3개의 개별 질문으로 분할.
        
        반드시 파이썬 List 형태의 문자열(예: ["질문1", "질문2"])만 출력하십시오. 절대 다른 설명은 붙이지 마십시오.
        
        사용자 질문: {query}
        """)
        
        chain = router_prompt | self.llm | StrOutputParser()
        result_str = chain.invoke({"query": query}).strip()
        
        try:
            sub_queries = eval(result_str)
            if not isinstance(sub_queries, list):
                sub_queries = [query]
        except:
            sub_queries = [query]
            
        print(f"\n[검색 쿼리 라우팅 결과]: {sub_queries}")
        return sub_queries

    def translate_query(self, query: str) -> str:
        trans_prompt = ChatPromptTemplate.from_template("""
        You are an elite AI compliance search engine. Your job is to convert the Korean user query into a highly optimized English keyword string for a vector database retrieval. 
        The database contains the EU AI Act and Korea AI Law in English.
        
        [DOMAIN KEYWORD MAPPING MATRIX]
        Analyze the user's query and APPEND the exact matched English keywords based on the following categories:
        
        1. HR / Employment (e.g., 인사, 채용, 평가, 승진, 근로자, 선발, 면접, 신입, 기수, 조직, 동아리, 업무):
           -> MUST APPEND: "Employment, workers management, recruitment, hiring, rights and obligations of individuals, Annex III, High-impact artificial intelligence, Korea AI Law"
        
        2. Biometrics / Emotion (e.g., 안면 인식, 홍채, 지문, 생체, 표정, 감정, 스트레스, 행동):
           -> MUST APPEND: "Biometric identification, emotion recognition, biometric information, Annex III, Prohibited AI practices, High-impact artificial intelligence, Korea AI Law"
           
        3. Education / Training (e.g., 교육, 입학, 성적, 학교, 학생, 시험, 학습, 훈련):
           -> MUST APPEND: "Education, vocational training, learning outcomes, evaluation of students, Annex III, High-impact artificial intelligence, Korea AI Law"
           
        4. Finance / Essential Services (e.g., 금융, 대출, 신용 평가, 보험, 복지, 응급, 구급차):
           -> MUST APPEND: "Essential private services, essential public services, credit score, loan screening, Annex III, High-impact artificial intelligence, Korea AI Law"
           
        5. Chatbots / Generative AI / Transparency (e.g., 챗봇, 안내, 생성형, 딥페이크, 고객센터, 대화):
           -> MUST APPEND: "Transparency obligations, interact with natural persons, Generative AI, Korea AI Law"
           
        6. Penalties / Obligations (e.g., 위험, 위반, 벌금, 과징금, 제재, 페널티, 의무, 조건, 규제):
           -> MUST APPEND: "High-Risk AI system, High-impact artificial intelligence, Sanctions, Penalties, administrative fines, responsibilities of business operators, Korea AI Law"
        
        [OUTPUT FORMAT]
        Return ONLY a single string of combined English keywords separated by spaces. Do not write sentences. Include the base translation of the query and the appended keywords.
        
        Korean Query: {query}
        """)
        chain = trans_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def retrieve_and_rerank_context(self, query: str, sub_queries: List[str], final_k: int = 15) -> str:
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embedding', 50, $query_vector)
        YIELD node AS chunk, score
        MATCH (parent)-[:HAS_CHUNK]->(chunk)
        
        OPTIONAL MATCH (parent)-[r1]->(t1)
        WHERE type(r1) <> 'HAS_CHUNK' AND type(r1) <> 'INCLUDES'
        
        OPTIONAL MATCH (t1)-[r2]->(t2)
        WHERE type(r2) IN ['MANDATED_FOR', 'PERMITS', 'PENALIZES_WITH', 'APPLIES_TO', 'LEADS_TO', 'ENCOMPASSES', 'SUPPLEMENTS']
        
        WITH chunk, score, parent, r1, t1, r2, t2,
             coalesce(t1.name, t1.level, t1.id, labels(t1)[0]) AS t1_base,
             CASE 
                WHEN 'Sanction' IN labels(t1) THEN ' [금액: ' + coalesce(t1['amount'], '명시안됨') + ', 유형: ' + coalesce(t1['type'], '분류안됨') + ', 상세: ' + coalesce(t1['description'], '없음') + ']'
                WHEN 'Requirement' IN labels(t1) OR 'Support' IN labels(t1) THEN ' [상세: ' + coalesce(t1['description'], '없음') + ']'
                WHEN 'TechCriterion' IN labels(t1) OR 'UsageCriterion' IN labels(t1) THEN ' [기준값: ' + coalesce(t1['threshold_value'], '') + ' ' + coalesce(t1['unit'], '') + ']'
                WHEN 'Concept' IN labels(t1) THEN ' [언어: ' + coalesce(t1['lang'], '') + ']'
                ELSE ''
             END AS t1_extra,
             CASE
                WHEN 'Concept' IN labels(t2) THEN ' [언어: ' + coalesce(t2['lang'], '') + ']'
                WHEN 'Requirement' IN labels(t2) OR 'Support' IN labels(t2) OR 'Sanction' IN labels(t2) THEN ' [상세: ' + coalesce(t2['description'], '없음') + ']'
                ELSE ''
             END AS t2_extra
             
        WITH chunk, score, parent,
             CASE 
                WHEN t1 IS NOT NULL AND t2 IS NULL THEN type(r1) + " -> " + t1_base + t1_extra
                WHEN t1 IS NOT NULL AND t2 IS NOT NULL THEN type(r1) + " -> " + t1_base + t1_extra + " => [" + type(r2) + " -> " + coalesce(t2.name, t2.level, t2.id, labels(t2)[0]) + t2_extra + "]"
                ELSE null
             END AS path_info

        RETURN 
            chunk.text AS chunk_text,
            labels(parent)[0] AS parent_type,
            coalesce(parent.id, parent.name, 'ID없음') AS parent_id,
            coalesce(parent.title, '') AS parent_title,
            collect(DISTINCT path_info) AS target_info
        """
        
        unique_chunks = {}
        translated_sq_list = [] 
        
        for sq in sub_queries:
            translated_query = self.translate_query(sq)
            translated_sq_list.append(translated_query) 
            
            query_vector = self.embeddings.embed_query(translated_query)
            results = self.graph.query(cypher_query, params={"query_vector": query_vector})
            
            for res in results:
                chunk_text = res['chunk_text']
                if chunk_text not in unique_chunks:
                    unique_chunks[chunk_text] = res

        if not unique_chunks:
            return "관련된 법적 근거를 찾을 수 없습니다."

        print(f"👉 1차 벡터 검색 완료: 총 {len(unique_chunks)}개의 조항 확보. 리랭킹 진행 중...")

        chunk_list = list(unique_chunks.values())
        combined_rerank_query = " ".join(translated_sq_list)
        pairs = [[combined_rerank_query, item['chunk_text']] for item in chunk_list]
        scores = self.reranker.predict(pairs)
        
        scored_results = sorted(zip(scores, chunk_list), key=lambda x: x[0], reverse=True)
        top_results = [item for score, item in scored_results[:final_k]]

        print(f"🎯 리랭킹 완료: 가장 적합 상위 {final_k}개 문서 추출.")

        formatted_context = []
        seen_chunks = set() # 중복 주입 방지용
        
        for res in top_results:
            p_type = res['parent_type']
            p_id = res['parent_id']
            p_title = res['parent_title']
            valid_targets = [t for t in res['target_info'] if t]
            t_info = "\n  - " + "\n  - ".join(valid_targets) if valid_targets else "추가 연결된 규제 없음"
            title_str = f" ({p_title})" if p_title else ""
            
            context_piece = (
                f"--- 출처: [{p_type}] {p_id}{title_str} ---\n"
                f"내용(Chunk): {res['chunk_text']}\n"
                f"관련 구조(Graph): {t_info}\n"
            )
            formatted_context.append(context_piece)
            seen_chunks.add(res['chunk_text'])

        print("🛡️ 필수 기초 조항(정의/의무) 하드캐리 주입 중...")
        essential_cypher = """
        MATCH (parent)-[:HAS_CHUNK]->(chunk)
        WHERE parent.id = 'Article 2' OR parent.id ENDS WITH '::Article 2'
           OR parent.id = 'Article 34' OR parent.id ENDS WITH '::Article 34'
           OR parent.id = 'Annex III' OR parent.id ENDS WITH '::Annex III'
        RETURN 
            chunk.text AS chunk_text,
            labels(parent)[0] AS parent_type,
            coalesce(parent.id, parent.name, 'ID없음') AS parent_id,
            coalesce(parent.title, '') AS parent_title
        """
        essential_results = self.graph.query(essential_cypher)
        
        for res in essential_results:
            if res['chunk_text'] not in seen_chunks:
                p_type = res['parent_type']
                p_id = res['parent_id']
                p_title = res['parent_title']
                title_str = f" ({p_title})" if p_title else ""
                context_piece = (
                    f"--- 출처: [{p_type}] {p_id}{title_str} ---\n"
                    f"내용(Chunk): {res['chunk_text']}\n"
                    f"관련 구조(Graph): 필수 정의 및 의무 조항 (시스템 강제 주입)\n"
                )
                formatted_context.append(context_piece)
                seen_chunks.add(res['chunk_text'])

        return "\n".join(formatted_context)

    def generate_answer(self, query: str, history: List[Dict] = None) -> Tuple[str, str]:
        
        history_text = ""
        if history:
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI(이전 답변)"
                history_text += f"[{role}]: {msg['content']}\n"
        else:
            history_text = "이전 대화 없음."

        sub_queries = self.analyze_and_route_query(query)
        context = self.retrieve_and_rerank_context(query, sub_queries, final_k=15)
        few_shot_examples = self.get_few_shot_examples(query, k=2)

        prompt = ChatPromptTemplate.from_template("""
        당신은 기업의 최고 AI 컴플라이언스 책임자(CCO)입니다.
        오직 아래 제공된 [Context]의 내용만을 사용하여 [사용자 질문]에 대해 최고 수준의 법률 컨설팅 답변을 작성하십시오. 당신이 학습한 사전 지식은 절대 사용하지 마십시오.

        [데이터 분석 및 출력 필수 지침]
        1. 내용 창조 금지: 제공된 데이터에 명시적으로 존재하지 않는 금액, 특정 의무 사항, 벌칙은 지어내지 마십시오.
        2. 출처 및 조항 번호 추출: 제공된 [Context]의 각 항목 시작 부분에 있는 `--- 출처: [종류] 조항_번호 (제목) ---` 포맷을 반드시 확인하십시오. 답변 본문과 요약의 '근거 조항' 작성 시, 이 출처 헤더에 적힌 법안명과 조항 번호를 명시해야 합니다.
        3. 모순된 정보 출력 금지: 정보가 존재하여 작성해 놓고 그 옆에 "(명시되지 않음)"이나 "(확인 불가)"라는 말을 동시에 적지 마십시오.
        4. 예시 복사 금지: [참고 예시]는 전개 방식만 참고하고, 내용은 절대 베끼지 마십시오.
        5. 시스템 용어 사용 금지: 화면에 "[Context]", "제공된 데이터" 같은 단어를 출력하지 마십시오. 
        6. 꼬리질문 대응 및 중복 설명 금지 (핵심 쟁점 타격): 사용자가 '비영리', '내부 사용', 특정 국가 등 방어 논리나 새로운 조건을 제시하며 꼬리질문을 던졌다면, 이전 답변에서 이미 설명한 원론적인 법 조항 내용(예: 고영향 AI의 일반적 정의 등)을 앵무새처럼 반복 나열하지 마십시오. 이전과 중복되는 설명은 과감히 생략하고, 대신 "사용자가 제시한 특정 조건(예: 수익 미창출)이 왜 법적으로 면책 사유가 되지 않는지" 그 심층적인 법리적 이유(예: 인공지능 법안의 규제 본질은 수익 창출 여부가 아니라 '개인 권리 침해의 중대성' 자체에 있음)를 정면으로 반박하는 데 집중하여 답변의 차별성을 극대화하십시오.
        7. 용어 통일: 한국의 법안을 지칭할 때는 반드시 '한국 인공지능기본법'으로 명칭을 통일하십시오.
        8. 제재 및 벌금 명시: 사용자의 질문에 "제재", "벌칙", "위반", "페널티" 등의 단어가 포함되어 있다면, 반드시 제공된 데이터에서 구체적인 제재 내용(예: 과징금 규모, 벌칙 조항)을 찾아 명시하십시오. 데이터에 세부 처벌 규정이 없다면 "현재 제공된 법안 내용에는 구체적인 제재 규모가 명시되어 있지 않으나, 법적 리스크 및 일반적인 행정 제재가 발생할 수 있습니다"라고 명확히 밝히십시오.
        [답변 구조 및 포맷 지침]
        아래 형식과 순서를 엄격히 지켜 답변하십시오.

        (자연스러운 산문 형태로 상세 답변을 작성하십시오. 단락 시작 부분에 '-', '*' 등의 기호를 붙이지 마십시오.)
        질문에 대한 '핵심 결론'을 가장 먼저 제시하는 두괄식으로 시작하십시오. 
        [억지 비유 금지 및 자연스러운 설명]: 규제 리스크를 설명할 때 조항 원문을 억지로 다른 분야에 비유하지 마십시오. 시스템이 '개인의 권리, 생계, 보상 등에 미칠 실질적인 영향'을 논리적으로 설명하십시오. 
        본문 설명 중에는 반드시 'Context의 출처 헤더'에서 파악한 관련 조항 번호를 자연스럽게 언급하십시오. 문장의 마지막에 '(출처: EU AI Act::Recital 57)'형태로 조항 번호를 언급하지 마십시오.
        
        [국가 미지정 시 대응 및 한국법 확장 해석 로직]
        대화 기록과 질문에 사용 국가가 특정되지 않았다면, 엄격한 'EU AI Act' 기준으로 설명한 뒤, 반드시 '한국 인공지능기본법' 관점도 비교하십시오. 
        - [논리적 서술 강제]: 한국 인공지능기본법을 설명할 때는 반드시 논리적 단계를 거치십시오. 1) 먼저 제2조(정의) 등을 근거로 해당 시스템이 개인의 권리와 의무에 중대한 영향을 미치는 '고영향 인공지능'에 해당함을 규명하고, 2) 그에 따라 기타 조항에 명시된 위험 관리 및 신뢰성 확보 의무를 준수해야 한다는 흐름으로 작성하십시오.
        - 만약 Context에 구체적 조항이 없다면, 포괄적인 기본 조항을 인용하며 다음과 같이 작성하십시오: "현재 한국 인공지능기본법에는 해당 시스템에 대한 구체적인 세부 요건이 명확히 명시되어 있지는 않습니다. 그러나 법안에 명시된 내용을 포괄적으로 확장하여 해석할 때, 이 시스템이 개인의 권리와 생계에 중대한 영향을 미칠 수 있으므로 규제 리스크가 존재할 수 있습니다."
        - [중요]: 한국법을 설명할 때, EU AI Act에만 있는 특정 의무(예: '인적 감독 할당')를 한국 인공지능기본법의 조항인 것처럼 섞어서 기술하지 마십시오. 두 법안의 내용을 철저히 분리하십시오.
        - 특정 국가가 명시되어 있다면 해당 국가 법률에 집중하십시오.
                                                  
        (줄바꿈 후 역질문 작성 - 필수)
        더욱 정확한 맞춤형 규제 리스크를 진단하기 위해, 추가로 확인이 필요한 사항은 다음과 같습니다.
        * (💡 대화 기록이나 질문에서 국가가 파악되지 않은 경우에만 1순위 질문: "해당 AI 시스템을 주로 어느 국가(시장)에서 서비스하실 계획인가요?" / 이미 특정 국가가 명시되었다면 이 질문은 절대 하지 마십시오.)
        * (구체적인 확인 사항 1)
        * (구체적인 확인 사항 2)

        (줄바꿈 후 요약 작성 - 필수)
        - 규제 대상: 확인된 핵심 요약
        - 핵심 의무: 확인된 핵심 요약 (정보 없을 시 "현행 규정에서는 확인되지 않음")
        - 위반 리스크: 확인된 핵심 요약 (정보 없을 시 "현행 규정에서는 확인되지 않음")
        - 근거 조항: 추출한 정확한 법안명 및 조항 번호. 정보가 없을 경우에만 "관련 법률상 명시되어 있지 않음" 기재.
        
        [이전 대화 기록]
        {history_text}

        [참고 예시]
        {few_shot_examples}
        
        [실제 법안 데이터 (Context)]
        {context}

        [사용자 질문 (Question)]
        {query}
        """)

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "history_text": history_text,
            "context": context, 
            "query": query,
            "few_shot_examples": few_shot_examples if few_shot_examples else "별도 예시 없음."
        })

        if answer.startswith("답변"):
            answer = answer.split("\n", 1)[-1].strip()

        return answer, context

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("AI Compliance Advanced RAG 엔진 가동 중...")
    rag = AIComplianceRAG()
    
    test_question = "인사 평가 시스템에 AI를 연동하면 고위험 AI에 해당하나요?"
    answer, context = rag.generate_answer(test_question)
    
    print("\n[최종 답변]")
    print(answer)