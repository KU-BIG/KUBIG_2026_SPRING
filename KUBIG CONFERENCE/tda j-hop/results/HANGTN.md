# hangtn — GTN이 고른 메타패스를 HAN에 주입 (선택 자동화 검증)

GTN attention 해독(c2_gtn 10 seed): 3 conv 모두 APVPA 1위(0.29~0.30), 채널 argmax 39/40 = APVPA·APVPA·APVPA.

| 조건 | 메타패스 | test macro-F1 |
|---|---|---|
| 기존 HAN (a) | APA·APTPA (수동) | 0.786±0.013 |
| han3 (수동+venue) | APA·APTPA·APVPA | 0.937±0.005 |
| **hangtn (GTN 선택)** | APVPA·APA (GTN attention top-2) | **0.938±0.003** (n=10) |
| (참고) RGCN (d) | 전 관계 | 0.934±0.004 |
