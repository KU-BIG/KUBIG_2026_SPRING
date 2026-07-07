# dblp venue-baseline 재계산 (han3 확장 + gh3)

dblp HAN 의 baseline 을 venue 포함 목록 {APA, APTPA, APVPA} 로 교체하고 HAN 계열 전부 재계산.
RGCN 은 원래 전 관계를 받아 변경 없음. seed 축소 지시(3개)로 일부 셀 n<10.

## 본표 HAN 행 (HANConv)

| (a) base | (b1) +noise | (b2) +mix | (c) +위상 |
|---|---|---|---|
| 0.937±0.005 (n=10) | 0.938±0.004 (n=10) | 0.956±0.008 (n=3) | 0.938±0.003 (n=10) |

## factorial HAN 행 (커스텀 gh3, 고정 GTN+PDGNN 위상)

| base | cat_real | cat_noise | cat_mix | gate_real | gate_noise | gate_mix |
|---|---|---|---|---|---|---|
| 0.936±0.006 (n=10) | 0.933±0.003 (n=10) | 0.928±0.006 (n=8) | 0.943±0.009 (n=3) | 0.934±0.005 (n=10) | 0.924±0.009 (n=10) | 0.936±0.007 (n=10) |

paired(n=10): cat_real−base −0.003(p=.38) · gate_real−base −0.002(p=.38) · gate_noise−base −0.012*(p=.027) · gate_mix−gate_real +0.002(p=.28)
판정: venue 를 채우면 dblp HAN 도 위상 기여 0 — 배달 이득 소멸, RGCN 과 동일 패턴.
