# 최종 통합 분석 — 위상 특징 × 이종 그래프 (NC · 주입 방식 · LP)

> **실험 전경** (random seed 10개, `*`=paired Wilcoxon p<0.05):
> ① NC 본캠페인 — 2백본(HAN/RGCN)×4조건(base/noise/class-mix/real)×7 데이터셋 (§1)
> ② 주입 factorial — 2백본×{주입 none/concat/gate × 내용 real/noise/mix}×3 데이터셋, 고정 manual 위상 (§2)
> ③ LP — L1(node-PI, 5 seed) + L2(pair-vicinity EPD, TLC-GNN식, 10 seed)×3 데이터셋 (§3)
> 검정: `paired_stats.md`(①) · `paired_stats2.md`(②③). 표: `SUMMARY.md` · `GATED.md` · `LP.md`.

## 1. NC — 위상을 feature 로 concat 하면 (본캠페인 요약)

- 유의한 이득은 HAN 위 dblp(+0.076\*)·aifb(+0.124\*) 뿐, RGCN 위에선 유의한 손해(f<d 4/4\*,
  imdb 는 noise 보다도 나쁨 −0.039\*).
- 이득의 분해: dblp = **관계 커버리지 누출**(HAN 메타패스에 없는 venue 관계가 GTN 채널로 주입 —
  같은 관계 위상만 쓰면 +0.004 로 소멸), aifb = §2.3 에서 재평가됨.
- class-mix 대조: **real 이 mix 를 이긴 곳 0/7** — per-node 정렬 기여 없음(concat 하에서).
- 상세 메커니즘·robustness·데이터셋 분석은 이전 판(§아카이브)과 동일 결론 — 여기선 새 실험이
  바꾼 부분을 중심으로 서술.

## 2. 주입 방식 factorial — concat vs gate (신규, 핵심)

같은 고정 위상(manual 채널·topo_seed 고정)을 **주입 방식만 바꿔** 비교. 모든 조건 동일 구현.

### 2.1 결과 (test macro-F1; 전체 표는 GATED.md)

| | base | cat_real | gate_real | gate_noise | 핵심 paired |
|---|---|---|---|---|---|
| RGCN acm | 0.930 | 0.929 | 0.936 | 0.924 | **gate_real−gate_mix +0.006\*** |
| RGCN dblp | 0.937 | 0.936 | 0.939 | 0.923 | gate_real−cat_real +0.003\* · gate_noise−base −0.013\* |
| RGCN aifb | 0.751 | 0.720 | **0.768** | 0.684 | gate−cat +0.049\* · **gate_real−gate_mix +0.029\*** · gate_noise −0.068\* |
| HAN acm | 0.885 | 0.883 | 0.884 | 0.867 | gate_noise−base −0.018\* |
| HAN dblp | 0.796 | 0.805 | 0.800 | 0.782 | gate_noise−base −0.014\* |
| HAN aifb | 0.710 | 0.585 | 0.700 | 0.585 | gate−cat +0.115\* · gate_noise −0.124\* |

### 2.2 발견

**F-A. 옛 (f) 하락의 재해석.** 고정 위상에선 cat_real≈base (acm/dblp, Δ −0.001) — 본캠페인
(f)<(d)의 큰 하락(−0.02~−0.075\*)은 concat 단독이 아니라 **concat × per-seed GTN 위상(불안정
특징)의 합작**이었다. 단 aifb 는 고정 위상으로도 concat 이 유해(−0.032/−0.125).

**F-B. 게이트 = 백본-일반 안전밸브, RGCN 한정 증폭기.** gate_real−cat_real 은 dblp\*·aifb\*
(RGCN)·aifb\*(HAN)에서 유의 양수 — 게이팅이 concat 의 해를 중화한다. gate_real−base 는 RGCN
에서 3/3 양수(+0.005~+0.017)나 개별 유의는 아님; HAN 에선 중립. → **"위상을 쓰려면 feature
concat 이 아니라 메시지 게이트로"** — PEGN(문헌)의 설계 선택이 실증적으로 정당화됨.

**F-C. 게이트도 내용물이 필요.** gate_noise−base 는 **6개 셀 중 5개 유의 음수**(−0.006~−0.124)
— 게이트는 쓰레기 신호에 방어적이지 않다. "게이팅 × 진짜 위상"의 조합만 안전/유익.

**F-D. 게이팅 하에서 per-node 정렬이 처음으로 유의해짐.** gate_real−gate_mix: RGCN acm
+0.006\*(p=.027)·aifb +0.029\*(p=.018). concat 패러다임에서 0/7 이던 정렬 기여가 게이트에선
2/3 에서 검출 — **위상의 instance-level 신호는 존재하되, 올바른 주입(게이트) 하에서만 관측
가능하며 크기는 작다(+0.006~+0.029).** (다중비교 감안 시 p≈.02~.03 은 경계적 — 재현 요망.)

### 2.3 aifb baseline 재평가 — "마지막 보루"의 붕괴

per-seed 분포 진단: **원조 HANConv(a)는 aifb 에서 10개 seed 중 9개가 정확히 0.438 에 고착**
(퇴화 최적점; 나머지 1개 0.572). 커스텀 attention 구현(gh_base)은 median 0.748(0.48~0.86)로
정상 학습. → 본캠페인의 "aifb 위상 이득 +0.124\*"의 실체는 위상의 정보 가치가 아니라 **고착된
baseline 을 흔들어 꺼내준 최적화 구출(optimization rescue)** 이다. 실제로 작동하는 baseline
(0.710) 위에선 concat 위상이 오히려 −0.125. **위상이 aggregate 성능을 올린 마지막 사례까지
비-위상 메커니즘으로 분해됨.**

## 3. LP — 위상 × link prediction (신규)

주 타겟-타겟 관계 엣지 예측, 고정 split, encoder=RGCN, val/test 엣지는 인접·위상에서 제거.

### 3.1 L1 (node-PI, encoder concat, 5 seed)
신호 없음: dblp 에서 **noise 가 real 을 이김**(c−b1 −0.015, win 20%) — 차원 정규화 효과뿐.
aifb 는 c−base 가 5/5 seed 일관 음수(−0.003). acm null.

### 3.2 L2 (pair-vicinity EPD, decoder concat = TLC-GNN 식, 10 seed)
- **aifb: real−base +0.002\*, real−noise +0.004\*** — 유의하지만 **크기가 무시 가능**
  (0.987→0.989, 천장 과제). 통계적 유의 ≠ 실질 기여의 교과서적 사례.
- acm: real−mix **−0.005** (mix 가 우세, p=.084) — pair-정렬 기여 없음, CN-bucket 수준 정보뿐.
- dblp: 전부 NS (real−base +0.006, p=.49).
- CN 휴리스틱 단독이 acm 0.79/dblp 0.72 — pair 위상이 CN 을 넘어 더하는 것이 거의 없음.

### 3.3 판정
**문헌이 위상에 가장 유리하다는 LP 에서도, 통제하면 실질 신호가 없다** — 본 프로젝트 선행
hetero-LP null 의 엄밀한 재확인. (aifb 천장·L1 5-seed 저검정력은 한계로 명시.)

## 4. 데이터셋 메커니즘 (최종판)

| 유형 | 데이터셋 | 실체 |
|---|---|---|
| A. 커버리지 누출 | dblp | 누락 관계(venue) 정보가 위상 채널 경유 주입 — 위상 고유 아님 |
| **B′. 최적화 구출 (재평가)** | aifb | ~~featureless 구조 인코딩~~ → **고착 baseline 구출** (§2.3). 작동하는 baseline 에선 concat 유해 |
| C. feature 포화 | acm·imdb | 무엇을 더해도 ≈0, imdb 는 concat-real 이 noise 보다 유해 |
| D. metric 붕괴 | mag·yelp | macro-F1 특성 (acc 로는 정상 학습) — Δ 비교만 유효 |
| E. 분산 지배 | freebase | 붕괴 seed(F1=0) 존재 — 비교 신뢰 낮음 |

## 5. 백본·메타패스 (불변 결론)

RGCN ≥ HAN accuracy 7/7 (paired 전부 유의). 병목 = 메타패스 *선택*(dblp venue −0.148).
메타패스의 가치는 성능이 아니라 해석성(채널 homophily 2.0× 실측). §2.3 은 여기에 "PyG HANConv
가 featureless 소형 그래프에서 고착할 수 있다"는 구현 각주를 추가한다.

## 6. 종합 결론 (최종)

1. **위상 특징의 내용**: NC 에선 class-level, LP 에선 CN-level 을 넘는 정보가 aggregate 지표에서
   검출되지 않음. concat 패러다임에서 per-node/pair 정렬 기여 = 0.
2. **주입 방식이 부호를 결정**: concat 은 무해~유해(불안정 특징·작동하는 baseline 에서 악화),
   **게이팅은 백본-일반 안전밸브**이며 RGCN 에선 소폭 증폭 — 단 진짜 위상일 때만(F-C).
3. **위상 신호는 "없다"가 아니라 "작고, 게이트 하에서만 보인다"**: gate_real>gate_mix\*(2/3)가
   instance-level 위상 신호의 첫 유의 증거 — 그러나 크기(+0.006~+0.029)가 백본 선택 효과
   (+0.03~+0.30)에 비해 한 자릿수 이상 작다.
4. **aggregate 이득으로 보였던 사례는 전부 비-위상 메커니즘으로 분해됨**: 커버리지 누출(dblp),
   최적화 구출(aifb). LP 도 구원 아님.
5. **실무 처방**: 이종 NC/LP 에서 성능이 목적이면 관계-완전 백본(RGCN 계열)이 우선. 위상을
   쓰겠다면 feature 가 아니라 **메시지 게이트**로, 그리고 진짜 위상으로. 기대 이득은 작다.

## 7. 한계

- 주입 factorial 은 3 데이터셋·고정 manual 위상(GTN per-seed 아님) — 옛 (f)와 특징이 다름(의도).
- 커스텀 conv(내부 일관성용)와 PyG 구현의 base 성능 차(aifb HANConv 고착 등) — 백본 구현 자체가
  교란 변수임을 §2.3 이 보여줌.
- 다중비교 미보정(경계 p 주의) · L1 5-seed · aifb LP 천장 · imdb/freebase 등 NC 일부 조건은
  협업 환경 실행값.
- per-node metric(margin/NLL)·η² 진단은 끝내 미수행 — F-D(게이트 하 정렬 신호)의 후속 검증처.

## 8. 참고

- [V] Yan et al., NeurIPS 2022 (PDGNN, arXiv:2201.12032) — NC 는 PEGN[56]·LP 는 TLC-GNN[50]에
  EPD 를 교체 주입(원문 §5.2, Table 2/3 확인)
- [V] Lee et al., ICML 2024 (arXiv:2402.04621) — within-class shuffle, A–X 의존성
- Zhao et al., AISTATS 2020 (PEGN) — persistence 로 메시지 재가중(게이팅) — §2 설계의 원형
- Yan et al., ICML 2021 (TLC-GNN) — pair-vicinity PH, decoder concat — §3-L2 설계의 원형
- Yun et al., NeurIPS 2019 (GTN) · Wang et al., WWW 2019 (HAN) · Schlichtkrull et al., ESWC 2018 (RGCN)
- Adams et al., JMLR 2017 (PI) · Strobl et al., 2008 (conditional permutation)
- 부속: `SUMMARY.md` `GATED.md` `LP.md` `paired_stats.md` `paired_stats2.md`
  `CLASS_WISE_MIXING.md` `f_rgcn_per_seed.md` `rerun_imdb_freebase.md` `DATASETS.md`
