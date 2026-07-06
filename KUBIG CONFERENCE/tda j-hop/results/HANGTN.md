# hangtn — GTN이 고른 메타패스를 HAN에 주입 (선택 자동화 검증)

GTN attention 해독(c2_gtn 10 seed): 3 conv 모두 APVPA 1위(0.29~0.30), 채널 argmax 39/40 = APVPA·APVPA·APVPA.

| 조건 | 메타패스 | test macro-F1 |
|---|---|---|
| 기존 HAN (a) | APA·APTPA (수동) | 0.786±0.013 |
| han3 (수동+venue) | APA·APTPA·APVPA | 0.937±0.005 |
| **hangtn (GTN 선택)** | APVPA·APA (GTN attention top-2) | **0.938±0.003** (n=10) |
| (참고) RGCN (d) | 전 관계 | 0.934±0.004 |

## 확장: imdb·mag (GTN top-2 되먹임, 10 seed)

| 데이터셋 | HAN (a, 수동) | hangtn (GTN 선택) | RGCN (d) |
|---|---|---|---|
| imdb | 0.438±0.009 (MDM·MAM) | **0.551±0.005** (MKM·MDM, n=10) | 0.636±0.004 |
| mag | 0.017±0.012 (PcP·PAP) | **0.034±0.020** (PFP·PcP, n=10) | 0.104±0.053 |

paired(hangtn−a): imdb macro +0.113 (win 100%, p=0.002) / acc +0.021 (p=0.002); mag macro +0.017 (win 80%, p=0.010) / acc +0.062 (win 90%, p=0.004).
격차 회수: dblp ~100%, imdb ~57%, mag(acc) ~43% — 잔여는 백본 커버리지 몫. 대조: acm·freebase attention 분산, aifb 구현 고착, yelp GTN 학습 실패.

## 음성 대조군: attention 분산/균등 4 ds (GTN top-2 되먹임, 10 seed)

| 데이터셋 | HAN (a, 수동) | hangtn (GTN top-2) | paired Δ | p |
|---|---|---|---|---|
| acm | 0.892±0.010 | 0.868±0.009 (PAP·PcP) | -0.027 (win 0%) | 0.00195 |
| freebase | 0.146±0.055 | 0.116±0.044 (BPB·BFB) | -0.030 (win 10%) | 0.0152 |
| aifb | 0.451±0.040 | 0.451±0.040 (r4·r5) | +0.000 (win 0%) | nan |
| yelp | 0.110±0.028 | 0.110±0.028 (BSB·BLB) | -0.000 (win 0%) | 0.18 |
