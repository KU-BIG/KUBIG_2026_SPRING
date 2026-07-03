# 클래스별 혼합 위상 Ablation

## 상태

- 생성 시각: 2026-07-03 01:44:15 UTC
- 완료 상태: `completed=140 missing=0 expected=140`
- 공식 run: 7 datasets x 2 backbones x 10 seeds = 140 runs
- 최종 지표가 모두 유한한 run: 138/140
- smoke-test seed `0`은 모든 요약에서 제외했다.
- 공식 seed: 312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219.

## 실험 설정

- Dataset: `acm`, `dblp`, `imdb`, `freebase`, `mag`, `aifb`, `yelp`.
- Backbone: HAN, RGCN.
- Topology source: `gtn`.
- Topology mode: `class_wise_mixing`.
- 결과 경로 패턴: `runs/class_wise_mixing/<dataset>__<backbone>__class_wise_mixing_s<seed>/metrics.json`.
- Topology cache는 class-wise mixing 이전의 실제 GTN-PDGNN 위상 특징만 저장한다. 이는 계산 재사용일 뿐이며 모델 또는 ablation 정의를 바꾸지 않는다.

## 동기와 해석

Class-wise mixing은 실제 GTN-PDGNN 위상 특징을 계산한 뒤, backbone feature와 concat하기 전에 각 노드의 위상 특징을 같은 class, 가능하면 같은 train/val/test split 안의 다른 노드 위상 특징으로 교체하는 구조적 대조군이다. class-level 위상 분포는 일부 보존하면서, 특정 노드와 특정 위상 특징 사이의 정렬만 깨뜨린다.

Random/noisy feature 대조군은 단순 차원 추가 효과를 검정한다. Class-wise mixing은 그보다 강한 반사실적 대조군으로, real GTN-PDGNN topology가 class-wise mixed topology보다 좋을 때 node-specific topology alignment가 의미 있다는 해석을 뒷받침한다.

## 현재 결과 요약표

| 데이터셋 | 백본 | n | 유한 지표 n | test_macro_f1 평균+/-표본표준편차 | test_accuracy 평균+/-표본표준편차 | val_macro_f1 평균+/-표본표준편차 | mixed ratio 평균+/-표본표준편차 | GTN attention NaN run 수 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| acm | HAN | 10 | 10 | 0.8990+/-0.0060 | 0.8974+/-0.0062 | 0.9281+/-0.0129 | 1.0000+/-0.0000 | 0 |
| acm | RGCN | 10 | 10 | 0.9232+/-0.0119 | 0.9223+/-0.0120 | 0.9442+/-0.0181 | 1.0000+/-0.0000 | 0 |
| dblp | HAN | 10 | 10 | 0.8776+/-0.0176 | 0.8818+/-0.0178 | 0.8825+/-0.0206 | 1.0000+/-0.0000 | 0 |
| dblp | RGCN | 10 | 10 | 0.9386+/-0.0048 | 0.9433+/-0.0044 | 0.9482+/-0.0053 | 1.0000+/-0.0000 | 0 |
| imdb | HAN | 10 | 10 | 0.4468+/-0.0205 | 0.7068+/-0.1152 | 0.4831+/-0.0158 | 0.9254+/-0.0000 | 10 |
| imdb | RGCN | 10 | 10 | 0.6341+/-0.0044 | 0.7912+/-0.0015 | 0.6807+/-0.0029 | 0.9254+/-0.0000 | 10 |
| freebase | HAN | 10 | 9 | 0.1618+/-0.0268 | 0.6239+/-0.0418 | 0.3832+/-0.0373 | 0.0990+/-0.0362 | 9 |
| freebase | RGCN | 10 | 9 | 0.2067+/-0.0564 | 0.6428+/-0.0326 | 0.3901+/-0.0745 | 0.0990+/-0.0362 | 9 |
| mag | HAN | 10 | 10 | 0.0186+/-0.0091 | 0.1420+/-0.0667 | 0.0336+/-0.0152 | 0.9839+/-0.0025 | 0 |
| mag | RGCN | 10 | 10 | 0.0881+/-0.0483 | 0.2829+/-0.1009 | 0.0883+/-0.0205 | 0.9839+/-0.0025 | 0 |
| aifb | HAN | 10 | 10 | 0.5379+/-0.1064 | 0.6778+/-0.0418 | 0.5018+/-0.0368 | 0.0771+/-0.0000 | 0 |
| aifb | RGCN | 10 | 10 | 0.7199+/-0.1335 | 0.7917+/-0.0810 | 0.6433+/-0.0403 | 0.0771+/-0.0000 | 0 |
| yelp | HAN | 10 | 10 | 0.0786+/-0.0238 | 0.7415+/-0.1530 | 0.0782+/-0.0233 | 0.9367+/-0.0000 | 0 |
| yelp | RGCN | 10 | 10 | 0.0666+/-0.0192 | 0.8271+/-0.0704 | 0.0676+/-0.0196 | 0.9367+/-0.0000 | 0 |

## 미완료 공식 Run

없음.

## Run별 상세표

| 데이터셋 | 백본 | seed | test_macro_f1 | test_accuracy | val_macro_f1 | mixed_nodes | unchanged_nodes | mixed_ratio | gtn_only_test_macro_f1 | gtn_only_test_accuracy | GTN attention NaN | 최종 지표 유한 | metrics 경로 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| acm | HAN | 312132 | 0.9008 | 0.8994 | 0.9270 | 3025 | 0 | 1.0000 | 0.8953 | 0.8933 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s312132/metrics.json` |
| acm | HAN | 238623 | 0.9108 | 0.9093 | 0.9333 | 3025 | 0 | 1.0000 | 0.9017 | 0.9004 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s238623/metrics.json` |
| acm | HAN | 792965 | 0.8984 | 0.8966 | 0.9038 | 3025 | 0 | 1.0000 | 0.8843 | 0.8820 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s792965/metrics.json` |
| acm | HAN | 15092 | 0.8962 | 0.8947 | 0.9191 | 3025 | 0 | 1.0000 | 0.8942 | 0.8928 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s15092/metrics.json` |
| acm | HAN | 661491 | 0.9007 | 0.8994 | 0.9333 | 3025 | 0 | 1.0000 | 0.9005 | 0.8990 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s661491/metrics.json` |
| acm | HAN | 588722 | 0.8886 | 0.8867 | 0.9428 | 3025 | 0 | 1.0000 | 0.9012 | 0.8999 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s588722/metrics.json` |
| acm | HAN | 825661 | 0.8929 | 0.8909 | 0.9278 | 3025 | 0 | 1.0000 | 0.9065 | 0.9051 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s825661/metrics.json` |
| acm | HAN | 500973 | 0.9024 | 0.9013 | 0.9187 | 3025 | 0 | 1.0000 | 0.8886 | 0.8862 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s500973/metrics.json` |
| acm | HAN | 88015 | 0.9022 | 0.9008 | 0.9497 | 3025 | 0 | 1.0000 | 0.8991 | 0.8975 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s88015/metrics.json` |
| acm | HAN | 251219 | 0.8965 | 0.8947 | 0.9259 | 3025 | 0 | 1.0000 | 0.8939 | 0.8919 | 아니오 | 예 | `runs/class_wise_mixing/acm__han__class_wise_mixing_s251219/metrics.json` |
| acm | RGCN | 312132 | 0.9395 | 0.9386 | 0.9545 | 3025 | 0 | 1.0000 | 0.8953 | 0.8933 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s312132/metrics.json` |
| acm | RGCN | 238623 | 0.9051 | 0.9042 | 0.9038 | 3025 | 0 | 1.0000 | 0.9017 | 0.9004 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s238623/metrics.json` |
| acm | RGCN | 792965 | 0.9197 | 0.9188 | 0.9416 | 3025 | 0 | 1.0000 | 0.8843 | 0.8820 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s792965/metrics.json` |
| acm | RGCN | 15092 | 0.9179 | 0.9169 | 0.9630 | 3025 | 0 | 1.0000 | 0.8942 | 0.8928 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s15092/metrics.json` |
| acm | RGCN | 661491 | 0.9065 | 0.9060 | 0.9331 | 3025 | 0 | 1.0000 | 0.9005 | 0.8990 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s661491/metrics.json` |
| acm | RGCN | 588722 | 0.9334 | 0.9325 | 0.9410 | 3025 | 0 | 1.0000 | 0.9012 | 0.8999 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s588722/metrics.json` |
| acm | RGCN | 825661 | 0.9328 | 0.9320 | 0.9630 | 3025 | 0 | 1.0000 | 0.9065 | 0.9051 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s825661/metrics.json` |
| acm | RGCN | 500973 | 0.9169 | 0.9155 | 0.9405 | 3025 | 0 | 1.0000 | 0.8886 | 0.8862 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s500973/metrics.json` |
| acm | RGCN | 88015 | 0.9258 | 0.9249 | 0.9394 | 3025 | 0 | 1.0000 | 0.8991 | 0.8975 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s88015/metrics.json` |
| acm | RGCN | 251219 | 0.9346 | 0.9339 | 0.9620 | 3025 | 0 | 1.0000 | 0.8939 | 0.8919 | 아니오 | 예 | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s251219/metrics.json` |
| dblp | HAN | 312132 | 0.8823 | 0.8859 | 0.8838 | 4057 | 0 | 1.0000 | 0.9222 | 0.9299 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s312132/metrics.json` |
| dblp | HAN | 238623 | 0.8769 | 0.8803 | 0.8983 | 4057 | 0 | 1.0000 | 0.9175 | 0.9254 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s238623/metrics.json` |
| dblp | HAN | 792965 | 0.8732 | 0.8785 | 0.8966 | 4057 | 0 | 1.0000 | 0.9191 | 0.9271 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s792965/metrics.json` |
| dblp | HAN | 15092 | 0.8651 | 0.8690 | 0.8781 | 4057 | 0 | 1.0000 | 0.9187 | 0.9268 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s15092/metrics.json` |
| dblp | HAN | 661491 | 0.9144 | 0.9180 | 0.9177 | 4057 | 0 | 1.0000 | 0.9143 | 0.9229 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s661491/metrics.json` |
| dblp | HAN | 588722 | 0.8918 | 0.8972 | 0.8939 | 4057 | 0 | 1.0000 | 0.9294 | 0.9356 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s588722/metrics.json` |
| dblp | HAN | 825661 | 0.8848 | 0.8894 | 0.8840 | 4057 | 0 | 1.0000 | 0.9213 | 0.9289 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s825661/metrics.json` |
| dblp | HAN | 500973 | 0.8749 | 0.8799 | 0.8550 | 4057 | 0 | 1.0000 | 0.9261 | 0.9320 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s500973/metrics.json` |
| dblp | HAN | 88015 | 0.8618 | 0.8658 | 0.8664 | 4057 | 0 | 1.0000 | 0.9256 | 0.9327 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s88015/metrics.json` |
| dblp | HAN | 251219 | 0.8504 | 0.8535 | 0.8513 | 4057 | 0 | 1.0000 | 0.9198 | 0.9275 | 아니오 | 예 | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s251219/metrics.json` |
| dblp | RGCN | 312132 | 0.9438 | 0.9479 | 0.9556 | 4057 | 0 | 1.0000 | 0.9222 | 0.9299 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s312132/metrics.json` |
| dblp | RGCN | 238623 | 0.9309 | 0.9352 | 0.9388 | 4057 | 0 | 1.0000 | 0.9175 | 0.9254 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s238623/metrics.json` |
| dblp | RGCN | 792965 | 0.9424 | 0.9465 | 0.9444 | 4057 | 0 | 1.0000 | 0.9191 | 0.9271 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s792965/metrics.json` |
| dblp | RGCN | 15092 | 0.9386 | 0.9433 | 0.9500 | 4057 | 0 | 1.0000 | 0.9187 | 0.9268 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s15092/metrics.json` |
| dblp | RGCN | 661491 | 0.9414 | 0.9461 | 0.9556 | 4057 | 0 | 1.0000 | 0.9143 | 0.9229 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s661491/metrics.json` |
| dblp | RGCN | 588722 | 0.9344 | 0.9405 | 0.9494 | 4057 | 0 | 1.0000 | 0.9294 | 0.9356 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s588722/metrics.json` |
| dblp | RGCN | 825661 | 0.9423 | 0.9465 | 0.9433 | 4057 | 0 | 1.0000 | 0.9213 | 0.9289 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s825661/metrics.json` |
| dblp | RGCN | 500973 | 0.9337 | 0.9394 | 0.9497 | 4057 | 0 | 1.0000 | 0.9261 | 0.9320 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s500973/metrics.json` |
| dblp | RGCN | 88015 | 0.9439 | 0.9479 | 0.9499 | 4057 | 0 | 1.0000 | 0.9256 | 0.9327 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s88015/metrics.json` |
| dblp | RGCN | 251219 | 0.9349 | 0.9398 | 0.9449 | 4057 | 0 | 1.0000 | 0.9198 | 0.9275 | 아니오 | 예 | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s251219/metrics.json` |
| imdb | HAN | 312132 | 0.4356 | 0.7434 | 0.4674 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s312132/metrics.json` |
| imdb | HAN | 238623 | 0.4365 | 0.7403 | 0.4867 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s238623/metrics.json` |
| imdb | HAN | 792965 | 0.4387 | 0.7413 | 0.4756 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s792965/metrics.json` |
| imdb | HAN | 15092 | 0.4431 | 0.7437 | 0.5012 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s15092/metrics.json` |
| imdb | HAN | 661491 | 0.4319 | 0.7422 | 0.4561 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s661491/metrics.json` |
| imdb | HAN | 588722 | 0.4375 | 0.7445 | 0.4757 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s588722/metrics.json` |
| imdb | HAN | 825661 | 0.4514 | 0.7443 | 0.5002 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s825661/metrics.json` |
| imdb | HAN | 500973 | 0.4566 | 0.7458 | 0.4836 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s500973/metrics.json` |
| imdb | HAN | 88015 | 0.5008 | 0.3788 | 0.5058 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s88015/metrics.json` |
| imdb | HAN | 251219 | 0.4355 | 0.7433 | 0.4787 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s251219/metrics.json` |
| imdb | RGCN | 312132 | 0.6429 | 0.7921 | 0.6776 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s312132/metrics.json` |
| imdb | RGCN | 238623 | 0.6319 | 0.7933 | 0.6801 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s238623/metrics.json` |
| imdb | RGCN | 792965 | 0.6290 | 0.7923 | 0.6816 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s792965/metrics.json` |
| imdb | RGCN | 15092 | 0.6279 | 0.7883 | 0.6774 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s15092/metrics.json` |
| imdb | RGCN | 661491 | 0.6308 | 0.7927 | 0.6837 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s661491/metrics.json` |
| imdb | RGCN | 588722 | 0.6340 | 0.7907 | 0.6831 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s588722/metrics.json` |
| imdb | RGCN | 825661 | 0.6363 | 0.7903 | 0.6822 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s825661/metrics.json` |
| imdb | RGCN | 500973 | 0.6348 | 0.7911 | 0.6765 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s500973/metrics.json` |
| imdb | RGCN | 88015 | 0.6370 | 0.7903 | 0.6850 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s88015/metrics.json` |
| imdb | RGCN | 251219 | 0.6362 | 0.7907 | 0.6803 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | 예 | 예 | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s251219/metrics.json` |
| freebase | HAN | 312132 | 0.1707 | 0.6368 | 0.3624 | 676 | 5324 | 0.1127 | 0.0025 | 0.0088 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s312132/metrics.json` |
| freebase | HAN | 238623 | 0.1475 | 0.6459 | 0.3444 | 650 | 5350 | 0.1083 | 0.0013 | 0.0045 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s238623/metrics.json` |
| freebase | HAN | 792965 | 0.1863 | 0.6403 | 0.3694 | 805 | 5195 | 0.1342 | 0.0010 | 0.0036 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s792965/metrics.json` |
| freebase | HAN | 15092 | 0.0000 | nan | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | nan | 예 | 아니오 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s15092/metrics.json` |
| freebase | HAN | 661491 | 0.1380 | 0.6407 | 0.3694 | 620 | 5380 | 0.1033 | 0.0020 | 0.0071 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s661491/metrics.json` |
| freebase | HAN | 588722 | 0.1383 | 0.5657 | 0.4037 | 637 | 5363 | 0.1062 | 0.0006 | 0.0022 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s588722/metrics.json` |
| freebase | HAN | 825661 | 0.1457 | 0.6052 | 0.4200 | 648 | 5352 | 0.1080 | 0.0025 | 0.0087 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s825661/metrics.json` |
| freebase | HAN | 500973 | 0.2057 | 0.5932 | 0.4178 | 707 | 5293 | 0.1178 | 0.0006 | 0.0020 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s500973/metrics.json` |
| freebase | HAN | 88015 | 0.1340 | 0.5831 | 0.3267 | 607 | 5393 | 0.1012 | 0.0006 | 0.0023 | 예 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s88015/metrics.json` |
| freebase | HAN | 251219 | 0.1898 | 0.7043 | 0.4349 | 589 | 5411 | 0.0982 | 0.1380 | 0.7067 | 아니오 | 예 | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s251219/metrics.json` |
| freebase | RGCN | 312132 | 0.1728 | 0.6499 | 0.4185 | 676 | 5324 | 0.1127 | 0.0025 | 0.0088 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s312132/metrics.json` |
| freebase | RGCN | 238623 | 0.1762 | 0.6793 | 0.4206 | 650 | 5350 | 0.1083 | 0.0013 | 0.0045 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s238623/metrics.json` |
| freebase | RGCN | 792965 | 0.1920 | 0.6475 | 0.3556 | 805 | 5195 | 0.1342 | 0.0010 | 0.0036 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s792965/metrics.json` |
| freebase | RGCN | 15092 | 0.0000 | nan | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | nan | 예 | 아니오 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s15092/metrics.json` |
| freebase | RGCN | 661491 | 0.2063 | 0.6572 | 0.4226 | 620 | 5380 | 0.1033 | 0.0020 | 0.0071 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s661491/metrics.json` |
| freebase | RGCN | 588722 | 0.1849 | 0.6036 | 0.3429 | 637 | 5363 | 0.1062 | 0.0006 | 0.0022 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s588722/metrics.json` |
| freebase | RGCN | 825661 | 0.1714 | 0.6117 | 0.3295 | 648 | 5352 | 0.1080 | 0.0025 | 0.0087 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s825661/metrics.json` |
| freebase | RGCN | 500973 | 0.3477 | 0.6493 | 0.5383 | 707 | 5293 | 0.1178 | 0.0006 | 0.0020 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s500973/metrics.json` |
| freebase | RGCN | 88015 | 0.2319 | 0.5968 | 0.2790 | 607 | 5393 | 0.1012 | 0.0006 | 0.0023 | 예 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s88015/metrics.json` |
| freebase | RGCN | 251219 | 0.1768 | 0.6899 | 0.4038 | 589 | 5411 | 0.0982 | 0.1380 | 0.7067 | 아니오 | 예 | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s251219/metrics.json` |
| mag | HAN | 312132 | 0.0141 | 0.1287 | 0.0325 | 5886 | 114 | 0.9810 | 0.0512 | 0.2632 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s312132/metrics.json` |
| mag | HAN | 238623 | 0.0100 | 0.1726 | 0.0328 | 5915 | 85 | 0.9858 | 0.0104 | 0.1726 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s238623/metrics.json` |
| mag | HAN | 792965 | 0.0317 | 0.1786 | 0.0294 | 5907 | 93 | 0.9845 | 0.0293 | 0.2302 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s792965/metrics.json` |
| mag | HAN | 15092 | 0.0109 | 0.0513 | 0.0746 | 5918 | 82 | 0.9863 | 0.0280 | 0.0769 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s15092/metrics.json` |
| mag | HAN | 661491 | 0.0183 | 0.1311 | 0.0253 | 5881 | 119 | 0.9802 | 0.0134 | 0.1844 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s661491/metrics.json` |
| mag | HAN | 588722 | 0.0344 | 0.1242 | 0.0281 | 5887 | 113 | 0.9812 | 0.0153 | 0.1438 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s588722/metrics.json` |
| mag | HAN | 825661 | 0.0142 | 0.1624 | 0.0303 | 5914 | 86 | 0.9857 | 0.0092 | 0.1624 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s825661/metrics.json` |
| mag | HAN | 500973 | 0.0259 | 0.2857 | 0.0344 | 5922 | 78 | 0.9870 | 0.0355 | 0.3233 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s500973/metrics.json` |
| mag | HAN | 88015 | 0.0174 | 0.1316 | 0.0321 | 5895 | 105 | 0.9825 | 0.0050 | 0.1513 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s88015/metrics.json` |
| mag | HAN | 251219 | 0.0087 | 0.0538 | 0.0170 | 5912 | 88 | 0.9853 | 0.0119 | 0.0860 | 아니오 | 예 | `runs/class_wise_mixing/mag__han__class_wise_mixing_s251219/metrics.json` |
| mag | RGCN | 312132 | 0.1027 | 0.2924 | 0.0989 | 5886 | 114 | 0.9810 | 0.0512 | 0.2632 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s312132/metrics.json` |
| mag | RGCN | 238623 | 0.0324 | 0.2965 | 0.1030 | 5915 | 85 | 0.9858 | 0.0104 | 0.1726 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s238623/metrics.json` |
| mag | RGCN | 792965 | 0.0784 | 0.4444 | 0.0815 | 5907 | 93 | 0.9845 | 0.0293 | 0.2302 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s792965/metrics.json` |
| mag | RGCN | 15092 | 0.2053 | 0.2308 | 0.1329 | 5918 | 82 | 0.9863 | 0.0280 | 0.0769 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s15092/metrics.json` |
| mag | RGCN | 661491 | 0.0642 | 0.2459 | 0.0685 | 5881 | 119 | 0.9802 | 0.0134 | 0.1844 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s661491/metrics.json` |
| mag | RGCN | 588722 | 0.0876 | 0.1961 | 0.0860 | 5887 | 113 | 0.9812 | 0.0153 | 0.1438 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s588722/metrics.json` |
| mag | RGCN | 825661 | 0.0735 | 0.4017 | 0.0969 | 5914 | 86 | 0.9857 | 0.0092 | 0.1624 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s825661/metrics.json` |
| mag | RGCN | 500973 | 0.1113 | 0.3910 | 0.0721 | 5922 | 78 | 0.9870 | 0.0355 | 0.3233 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s500973/metrics.json` |
| mag | RGCN | 88015 | 0.0870 | 0.1579 | 0.0797 | 5895 | 105 | 0.9825 | 0.0050 | 0.1513 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s88015/metrics.json` |
| mag | RGCN | 251219 | 0.0382 | 0.1720 | 0.0632 | 5912 | 88 | 0.9853 | 0.0119 | 0.0860 | 아니오 | 예 | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s251219/metrics.json` |
| aifb | HAN | 312132 | 0.5314 | 0.6389 | 0.5511 | 175 | 2095 | 0.0771 | 0.3547 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s312132/metrics.json` |
| aifb | HAN | 238623 | 0.5794 | 0.6944 | 0.5203 | 175 | 2095 | 0.0771 | 0.3020 | 0.5556 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s238623/metrics.json` |
| aifb | HAN | 792965 | 0.5742 | 0.7222 | 0.4986 | 175 | 2095 | 0.0771 | 0.2963 | 0.5278 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s792965/metrics.json` |
| aifb | HAN | 15092 | 0.6346 | 0.7222 | 0.5511 | 175 | 2095 | 0.0771 | 0.3000 | 0.5278 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s15092/metrics.json` |
| aifb | HAN | 661491 | 0.4102 | 0.6944 | 0.5203 | 175 | 2095 | 0.0771 | 0.3393 | 0.5000 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s661491/metrics.json` |
| aifb | HAN | 588722 | 0.4942 | 0.6667 | 0.4913 | 175 | 2095 | 0.0771 | 0.3430 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s588722/metrics.json` |
| aifb | HAN | 825661 | 0.5170 | 0.6944 | 0.4318 | 175 | 2095 | 0.0771 | 0.3590 | 0.6389 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s825661/metrics.json` |
| aifb | HAN | 500973 | 0.7007 | 0.6944 | 0.4633 | 175 | 2095 | 0.0771 | 0.3494 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s500973/metrics.json` |
| aifb | HAN | 88015 | 0.6000 | 0.6667 | 0.4986 | 175 | 2095 | 0.0771 | 0.3054 | 0.5556 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s88015/metrics.json` |
| aifb | HAN | 251219 | 0.3370 | 0.5833 | 0.4913 | 175 | 2095 | 0.0771 | 0.3758 | 0.6667 | 아니오 | 예 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s251219/metrics.json` |
| aifb | RGCN | 312132 | 0.8182 | 0.8333 | 0.7143 | 175 | 2095 | 0.0771 | 0.3547 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s312132/metrics.json` |
| aifb | RGCN | 238623 | 0.8355 | 0.8333 | 0.6271 | 175 | 2095 | 0.0771 | 0.3020 | 0.5556 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s238623/metrics.json` |
| aifb | RGCN | 792965 | 0.8979 | 0.8889 | 0.6271 | 175 | 2095 | 0.0771 | 0.2963 | 0.5278 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s792965/metrics.json` |
| aifb | RGCN | 15092 | 0.7031 | 0.7778 | 0.6717 | 175 | 2095 | 0.0771 | 0.3000 | 0.5278 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s15092/metrics.json` |
| aifb | RGCN | 661491 | 0.7486 | 0.8056 | 0.6370 | 175 | 2095 | 0.0771 | 0.3393 | 0.5000 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s661491/metrics.json` |
| aifb | RGCN | 588722 | 0.4742 | 0.6667 | 0.5716 | 175 | 2095 | 0.0771 | 0.3430 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s588722/metrics.json` |
| aifb | RGCN | 825661 | 0.8104 | 0.8333 | 0.6370 | 175 | 2095 | 0.0771 | 0.3590 | 0.6389 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s825661/metrics.json` |
| aifb | RGCN | 500973 | 0.5936 | 0.6389 | 0.6935 | 175 | 2095 | 0.0771 | 0.3494 | 0.6111 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s500973/metrics.json` |
| aifb | RGCN | 88015 | 0.7386 | 0.8611 | 0.6271 | 175 | 2095 | 0.0771 | 0.3054 | 0.5556 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s88015/metrics.json` |
| aifb | RGCN | 251219 | 0.5790 | 0.7778 | 0.6271 | 175 | 2095 | 0.0771 | 0.3758 | 0.6667 | 아니오 | 예 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s251219/metrics.json` |
| yelp | HAN | 312132 | 0.0585 | 0.8219 | 0.0598 | 5137 | 347 | 0.9367 | 0.1220 | 0.3666 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s312132/metrics.json` |
| yelp | HAN | 238623 | 0.0732 | 0.8154 | 0.0723 | 5137 | 347 | 0.9367 | 0.1105 | 0.6303 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s238623/metrics.json` |
| yelp | HAN | 792965 | 0.0503 | 0.8757 | 0.0511 | 5137 | 347 | 0.9367 | 0.1616 | 0.4389 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s792965/metrics.json` |
| yelp | HAN | 15092 | 0.1244 | 0.6481 | 0.1222 | 5137 | 347 | 0.9367 | 0.1186 | 0.5765 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s15092/metrics.json` |
| yelp | HAN | 661491 | 0.0825 | 0.5644 | 0.0799 | 5137 | 347 | 0.9367 | 0.1146 | 0.7003 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s661491/metrics.json` |
| yelp | HAN | 588722 | 0.0504 | 0.8758 | 0.0511 | 5137 | 347 | 0.9367 | 0.1400 | 0.5393 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s588722/metrics.json` |
| yelp | HAN | 825661 | 0.1053 | 0.4096 | 0.1090 | 5137 | 347 | 0.9367 | 0.0877 | 0.6662 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s825661/metrics.json` |
| yelp | HAN | 500973 | 0.0845 | 0.7906 | 0.0820 | 5137 | 347 | 0.9367 | 0.0825 | 0.5087 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s500973/metrics.json` |
| yelp | HAN | 88015 | 0.0683 | 0.8341 | 0.0690 | 5137 | 347 | 0.9367 | 0.1161 | 0.5737 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s88015/metrics.json` |
| yelp | HAN | 251219 | 0.0888 | 0.7795 | 0.0853 | 5137 | 347 | 0.9367 | 0.1281 | 0.5892 | 아니오 | 예 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s251219/metrics.json` |
| yelp | RGCN | 312132 | 0.1012 | 0.6839 | 0.1052 | 5137 | 347 | 0.9367 | 0.1220 | 0.3666 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s312132/metrics.json` |
| yelp | RGCN | 238623 | 0.0877 | 0.7826 | 0.0890 | 5137 | 347 | 0.9367 | 0.1105 | 0.6303 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s238623/metrics.json` |
| yelp | RGCN | 792965 | 0.0674 | 0.8304 | 0.0672 | 5137 | 347 | 0.9367 | 0.1616 | 0.4389 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s792965/metrics.json` |
| yelp | RGCN | 15092 | 0.0509 | 0.8759 | 0.0511 | 5137 | 347 | 0.9367 | 0.1186 | 0.5765 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s15092/metrics.json` |
| yelp | RGCN | 661491 | 0.0510 | 0.8774 | 0.0518 | 5137 | 347 | 0.9367 | 0.1146 | 0.7003 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s661491/metrics.json` |
| yelp | RGCN | 588722 | 0.0512 | 0.8778 | 0.0520 | 5137 | 347 | 0.9367 | 0.1400 | 0.5393 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s588722/metrics.json` |
| yelp | RGCN | 825661 | 0.0504 | 0.8759 | 0.0514 | 5137 | 347 | 0.9367 | 0.0877 | 0.6662 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s825661/metrics.json` |
| yelp | RGCN | 500973 | 0.0623 | 0.8627 | 0.0616 | 5137 | 347 | 0.9367 | 0.0825 | 0.5087 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s500973/metrics.json` |
| yelp | RGCN | 88015 | 0.0892 | 0.7323 | 0.0885 | 5137 | 347 | 0.9367 | 0.1161 | 0.5737 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s88015/metrics.json` |
| yelp | RGCN | 251219 | 0.0542 | 0.8718 | 0.0577 | 5137 | 347 | 0.9367 | 0.1281 | 0.5892 | 아니오 | 예 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s251219/metrics.json` |

## 진단 / Caveat

- GTN attention NaN이 있는 그룹: imdb/HAN 10/10; imdb/RGCN 10/10; freebase/HAN 9/10; freebase/RGCN 9/10.
- 최종 지표 NaN 예외: 2/140 run: `runs/class_wise_mixing/freebase__han__class_wise_mixing_s15092/metrics.json`, `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s15092/metrics.json`.
- Freebase GTN attention caveat: Freebase 20개 run 중 18/20개에서 `gtn_attentions`에 NaN이 있다. 최종 지표는 18/20개 run에서 유한하다.
- Freebase mixed ratio: 0.0990+/-0.0353.
- IMDB도 저장된 `gtn_attentions`에는 NaN이 있지만 최종 지표는 유한하다. 이 값은 숨기지 않고 attention artifact 진단 caveat로 보고한다.
- 같은 class 및 같은 split 안의 group이 작으면 class-wise mixing이 약해질 수 있다. singleton group은 그대로 남으며 `unchanged_nodes`와 mixed ratio에 반영된다.
- MAG는 class 수가 많고 subsampled node를 사용하므로 macro-F1 절대값이 낮을 수 있다. Yelp는 HNE featureless multilabel 설정이라 macro-F1을 accuracy 및 같은 dataset 안의 조건 간 차이와 함께 해석해야 한다.

## 재현 명령

완료 run 수 확인:

```bash
find runs/class_wise_mixing -name metrics.json | grep -Ev '_s0/metrics.json$' | wc -l
```

누락 run 확인:

```bash
python - <<'PY'
from pathlib import Path
datasets=['acm','dblp','imdb','freebase','mag','aifb','yelp']
backbones=['han','rgcn']
seeds=[312132,238623,792965,15092,661491,588722,825661,500973,88015,251219]
root=Path('runs/class_wise_mixing')
missing=[]
for ds in datasets:
  for bb in backbones:
    for s in seeds:
      p=root/f'{ds}__{bb}__class_wise_mixing_s{s}'/'metrics.json'
      if not p.exists(): missing.append(str(p))
print(f'completed={140-len(missing)} missing={len(missing)} expected=140')
print('\n'.join(missing))
PY
```

요약 재생성:

```bash
python experiments/summarize_class_wise_mixing.py
```

재실행/이어 실행:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate tda
cd /workspace/TDA
bash experiments/run_class_wise_mixing.sh
```

이미 `metrics.json`이 있는 run은 `experiments/run_class_wise_mixing.sh`에서 skip된다.
