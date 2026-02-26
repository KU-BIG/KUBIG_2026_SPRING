import os
import sys

# dotenv 미설치 시 안내 메시지 출력 및 예외 처리
try:
    from dotenv import load_dotenv
except ImportError:
    print("오류: 'python-dotenv' 모듈을 찾을 수 없습니다.")
    print("해결 방법: 터미널에서 'pip install python-dotenv'를 실행하여 패키지를 설치해주세요.")
    sys.exit(1)

from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 환경 변수 로드 (.env)
load_dotenv()

class GraphDBConnection:
    """
    Neo4j 및 OpenAI 연결 설정을 관리하는 클래스
    """
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError("Neo4j 환경 변수(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)가 설정되지 않았습니다.")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    def get_graph(self) -> Neo4jGraph:
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )

    def get_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=self.openai_api_key
        )