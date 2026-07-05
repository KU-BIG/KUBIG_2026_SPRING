# imdb·freebase 재실행 결과 (GTN NaN 버그픽스 후)

이전 run 은 GTN NaN→위상 0벡터(silent failure)여서 무효 — `_gtn_norm` clamp 픽스 후 재실행.
조건: c2_gtn=(c) HAN+위상, b2_mix=(b2), e2_mix=(e2), f_rgcn=(f). test Macro-F1 / accuracy, mean±std (10 seed).

| 데이터셋 | 조건 | macro-F1 | accuracy | n | GTN-attn NaN run |
|---|---|---|---|---|---|
| imdb | c2_gtn | 0.402±0.015 | 0.735±0.003 | 10 | 0 |
| imdb | b2_mix | 0.402±0.013 | 0.736±0.003 | 10 | 0 |
| imdb | e2_mix | 0.575±0.033 | 0.776±0.009 | 10 | 0 |
| imdb | f_rgcn | 0.566±0.035 | 0.775±0.010 | 10 | 0 |
| freebase | c2_gtn | 0.138±0.049 | 0.607±0.042 | 10 | 0 |
| freebase | b2_mix | 0.141±0.060 | 0.609±0.041 | 10 | 0 |
| freebase | e2_mix | 0.170±0.077 | 0.588±0.063 | 10 | 0 |
| freebase | f_rgcn | 0.169±0.059 | 0.601±0.052 | 10 | 0 |
