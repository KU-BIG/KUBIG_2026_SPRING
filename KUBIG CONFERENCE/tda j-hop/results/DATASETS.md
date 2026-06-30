# 데이터셋 특징 정리 (14개) + 간소화 추천

모든 수치는 `experiments/dataset_stats.py`로 **실제 로드해서 측정**한 값(`experiments/dataset_stats.json`).
노드/엣지는 이종그래프의 **전체 타입 합**, 타겟노드는 분류 대상 타입.

## 1. 전체 14개 데이터셋 특징

| 데이터셋 | 도메인 | 타겟노드 | 노드타입 | 총노드 | 엣지타입 | 총엣지 | 클래스 | feature | 라벨 |
|---|---|---|---|---|---|---|---|---|---|
| acm | 학술/인용 | paper 3,025 | 4 | 10,942 | 8 | 547,872 | 3 | **1,902** | 단일 |
| dblp | 학술/인용 | author 4,057 | 4 | 26,128 | 6 | 239,566 | 4 | 334 | 단일 |
| imdb | 영화 | movie 4,932 | 4 | 21,420 | 6 | 86,642 | 5 | **3,489** | **멀티(5)** |
| freebase | 지식그래프 | book 40,402 | **8** | 180,098 | **36** | 1,057,688 | 7 | featureless | 단일 |
| aminer | 학술(대형) | author 14,949 | 3 | 33,554 | 4 | 99,442 | 8 | featureless | 단일 |
| mag | 학술(초대형) | paper 11,998 | 4 | **1,164,525** | 4 | 1,488,139 | **349** | 128 | 단일 |
| dblp_pyg | 학술(PyG판) | author 4,057 | 4 | 26,128 | 6 | 239,566 | 4 | 334 | 단일 |
| imdb_pyg | 영화(단일판) | movie 4,278 | 3 | 11,616 | 4 | 34,212 | 3 | 3,066 | 단일 |
| aifb | RDF/연구기관 | entity 2,270 | 1 | 2,270 | 6 | 23,290 | 4 | featureless | 단일 |
| mutag | RDF/화학 | entity 4,000 | 1 | 4,000 | 6 | 7,998 | 2 | featureless | 단일 |
| bgs | RDF/지질 | entity 3,993 | 1 | 3,993 | 6 | 13,568 | 2 | featureless | 단일 |
| am | RDF/박물관 | entity 5,967 | 1 | 5,967 | 6 | 84,838 | 11 | featureless | 단일 |
| pubmed | 생의학(HNE) | t1 6,889 | 4 | 49,835 | 10 | 202,290 | 8 | 200 | 단일 |
| yelp | business(HNE) | t0 5,484 | 4 | 29,530 | 4 | **3,878,837** | 16 | featureless | **멀티(16)** |

**주의(subsample):** 위는 *로드 시* 측정값입니다. 대형(freebase/mag/aminer/pubmed/yelp)은 dense GTN/HKS 제약상
파이프라인에서 **타겟노드 ≤6000(`max_target_nodes`)** 로 추가 subsample되고, RDF/HNE 로더는 자체 cap을 적용합니다.

## 2. 다양성 축 분석

- **도메인**: 학술(acm·dblp·dblp_pyg·aminer·mag) / 영화(imdb·imdb_pyg) / 지식그래프(freebase) / RDF(aifb·mutag·bgs·am) / 생의학(pubmed) / business(yelp)
- **feature 강도** (위상 효용을 가르는 핵심 축): 강(acm 1902·imdb 3489·imdb_pyg 3066) · 중(dblp 334·pubmed 200·mag 128) · **featureless**(freebase·aminer·aifb·mutag·bgs·am·yelp)
- **라벨**: 단일 / **멀티**(imdb·yelp)
- **규모**: 소형(aifb 2,270) → **초대형**(mag 1.16M 노드)
- **노드타입 수**: 단일타입(RDF 4종) → **많음**(freebase 8)
- **밀집도**: 희소(mutag 7,998 엣지) → **초밀집**(yelp 3.88M, acm 547K)

**중복(간소화 대상):**
- **dblp ≡ dblp_pyg** — 측정값 *완전 동일*(author 4,057 / 26,128 / 239,566 / feat 334 / 4클래스). → 하나만.
- **RDF 4종(aifb·mutag·bgs·am)** — 전부 featureless·단일노드타입·6관계로 성격 유사. → 1~2개면 충분.
- **학술 5종(acm·dblp·dblp_pyg·aminer·mag)** — 과다. → 2~3개로.
- **imdb vs imdb_pyg** — 같은 영화지만 imdb=멀티라벨(5), imdb_pyg=단일(3). 둘 중 imdb가 task 다양성↑.

## 3. 추천 간소화 subset (7개, 다양성 최대화)

| # | 데이터셋 | 커버하는 축 |
|---|---|---|
| 1 | **acm** | 학술 · **feature 강(1902)** · 단일 · dense → "feature 강" 대표(위상 효과 ≈0 기준선) |
| 2 | **dblp** | 학술 · **feature 약(334)** · 단일 → "feature 약" 대표(위상 Δ↑ 관측) |
| 3 | **imdb** | 영화 · feature 강 · **멀티라벨** → 도메인 + task 다양성 |
| 4 | **freebase** | **지식그래프** · **featureless** · **노드타입 8 · 36관계** → featureless+복잡구조 |
| 5 | **mag** | 학술 · **초대형(1.16M·349클래스)** → 규모 극단 |
| 6 | **aifb** | **RDF** · featureless · 단일노드타입 · 소형 → RDF·다관계 대표 |
| 7 | **yelp** | **business** · featureless · **멀티라벨** · **초밀집(3.9M 엣지)** → 도메인+밀집도 |

이 7개로 **도메인 5종, feature 강/약/없음, 단일/멀티라벨, 소형~초대형, 단일~다중 노드타입, 희소~초밀집**을 모두 커버합니다.

**제외(중복/유사):** dblp_pyg(=dblp), imdb_pyg(≈imdb 단일판), aminer(학술 featureless—freebase·mag로 커버), mutag·bgs·am(RDF—aifb로 커버).

**옵션:** 생의학 도메인을 꼭 넣으려면 **+pubmed**(→8개). featureless-학술을 따로 보고 싶으면 **+aminer**.
