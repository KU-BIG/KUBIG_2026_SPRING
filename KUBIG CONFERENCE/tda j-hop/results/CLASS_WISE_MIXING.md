# Class-wise Mixing Topology Ablation

## Status

Generated: 2026-07-02 12:27:35 UTC

Status: **in progress**. Completed official runs: **100/140**; finite final metrics: **98/100**.

Smoke-test seed `0` is excluded from all summaries. Official seeds: 312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219.

Completed dataset/backbone pairs:

- acm / HAN: n=10/10
- acm / RGCN: n=10/10
- dblp / HAN: n=10/10
- dblp / RGCN: n=10/10
- imdb / HAN: n=10/10
- imdb / RGCN: n=10/10
- freebase / HAN: n=10/10
- freebase / RGCN: n=10/10
- mag / HAN: n=10/10
- mag / RGCN: n=10/10

Incomplete dataset/backbone pairs:

- aifb / HAN: n=0/10
- aifb / RGCN: n=0/10
- yelp / HAN: n=0/10
- yelp / RGCN: n=0/10

## Motivation

Class-wise mixing is a structured topology control inspired by CFH-style within-class feature mixing. The run first computes the real GTN-PDGNN topology feature, then before concatenation replaces each node's topology feature with another node's topology feature from the same class, preferably within the same train/val/test split.

Random/noisy feature controls test whether simply adding feature dimensions helps. Class-wise mixing is a stronger counterfactual for topology features: it preserves class-level topology-feature distribution to some extent while breaking node-specific topology-feature alignment. If real GTN-PDGNN topology outperforms class-wise mixed topology, that supports the claim that the topology feature carries meaningful node-specific signal.

## Experimental Setup

- Datasets: `acm`, `dblp`, `imdb`, `freebase`, `mag`, `aifb`, `yelp`.
- Backbones: HAN and RGCN.
- Topology source: `gtn`.
- Topology mode: `class_wise_mixing`.
- Expected total: 7 datasets x 2 backbones x 10 seeds = 140 runs.
- Result path pattern: `runs/class_wise_mixing/<dataset>__<backbone>__class_wise_mixing_s<seed>/metrics.json`.
- The topology cache stores real GTN-PDGNN topology features before class-wise mixing. It is computation reuse only and does not change the model or ablation definition. Class-wise mixing is applied after loading or computing the real topology feature.

## Current Results Table

| dataset | backbone | n | finite n | test_macro_f1 mean+/-std | test_accuracy mean+/-std | val_macro_f1 mean+/-std | mixed ratio mean+/-std | NaN GTN attention runs |
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

## Missing Official Runs

| dataset | backbone | seed | expected metrics path |
|---|---|---:|---|
| aifb | HAN | 312132 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s312132/metrics.json` |
| aifb | HAN | 238623 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s238623/metrics.json` |
| aifb | HAN | 792965 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s792965/metrics.json` |
| aifb | HAN | 15092 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s15092/metrics.json` |
| aifb | HAN | 661491 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s661491/metrics.json` |
| aifb | HAN | 588722 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s588722/metrics.json` |
| aifb | HAN | 825661 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s825661/metrics.json` |
| aifb | HAN | 500973 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s500973/metrics.json` |
| aifb | HAN | 88015 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s88015/metrics.json` |
| aifb | HAN | 251219 | `runs/class_wise_mixing/aifb__han__class_wise_mixing_s251219/metrics.json` |
| aifb | RGCN | 312132 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s312132/metrics.json` |
| aifb | RGCN | 238623 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s238623/metrics.json` |
| aifb | RGCN | 792965 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s792965/metrics.json` |
| aifb | RGCN | 15092 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s15092/metrics.json` |
| aifb | RGCN | 661491 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s661491/metrics.json` |
| aifb | RGCN | 588722 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s588722/metrics.json` |
| aifb | RGCN | 825661 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s825661/metrics.json` |
| aifb | RGCN | 500973 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s500973/metrics.json` |
| aifb | RGCN | 88015 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s88015/metrics.json` |
| aifb | RGCN | 251219 | `runs/class_wise_mixing/aifb__rgcn__class_wise_mixing_s251219/metrics.json` |
| yelp | HAN | 312132 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s312132/metrics.json` |
| yelp | HAN | 238623 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s238623/metrics.json` |
| yelp | HAN | 792965 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s792965/metrics.json` |
| yelp | HAN | 15092 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s15092/metrics.json` |
| yelp | HAN | 661491 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s661491/metrics.json` |
| yelp | HAN | 588722 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s588722/metrics.json` |
| yelp | HAN | 825661 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s825661/metrics.json` |
| yelp | HAN | 500973 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s500973/metrics.json` |
| yelp | HAN | 88015 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s88015/metrics.json` |
| yelp | HAN | 251219 | `runs/class_wise_mixing/yelp__han__class_wise_mixing_s251219/metrics.json` |
| yelp | RGCN | 312132 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s312132/metrics.json` |
| yelp | RGCN | 238623 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s238623/metrics.json` |
| yelp | RGCN | 792965 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s792965/metrics.json` |
| yelp | RGCN | 15092 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s15092/metrics.json` |
| yelp | RGCN | 661491 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s661491/metrics.json` |
| yelp | RGCN | 588722 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s588722/metrics.json` |
| yelp | RGCN | 825661 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s825661/metrics.json` |
| yelp | RGCN | 500973 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s500973/metrics.json` |
| yelp | RGCN | 88015 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s88015/metrics.json` |
| yelp | RGCN | 251219 | `runs/class_wise_mixing/yelp__rgcn__class_wise_mixing_s251219/metrics.json` |

## Per-run Table

| dataset | backbone | seed | test_macro_f1 | test_accuracy | val_macro_f1 | mixed_nodes | unchanged_nodes | mixed_ratio | gtn_only_test_macro_f1 | gtn_only_test_accuracy | nan_gtn_attention | final_metrics_finite | metrics path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| acm | HAN | 312132 | 0.9008 | 0.8994 | 0.9270 | 3025 | 0 | 1.0000 | 0.8953 | 0.8933 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s312132/metrics.json` |
| acm | HAN | 238623 | 0.9108 | 0.9093 | 0.9333 | 3025 | 0 | 1.0000 | 0.9017 | 0.9004 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s238623/metrics.json` |
| acm | HAN | 792965 | 0.8984 | 0.8966 | 0.9038 | 3025 | 0 | 1.0000 | 0.8843 | 0.8820 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s792965/metrics.json` |
| acm | HAN | 15092 | 0.8962 | 0.8947 | 0.9191 | 3025 | 0 | 1.0000 | 0.8942 | 0.8928 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s15092/metrics.json` |
| acm | HAN | 661491 | 0.9007 | 0.8994 | 0.9333 | 3025 | 0 | 1.0000 | 0.9005 | 0.8990 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s661491/metrics.json` |
| acm | HAN | 588722 | 0.8886 | 0.8867 | 0.9428 | 3025 | 0 | 1.0000 | 0.9012 | 0.8999 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s588722/metrics.json` |
| acm | HAN | 825661 | 0.8929 | 0.8909 | 0.9278 | 3025 | 0 | 1.0000 | 0.9065 | 0.9051 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s825661/metrics.json` |
| acm | HAN | 500973 | 0.9024 | 0.9013 | 0.9187 | 3025 | 0 | 1.0000 | 0.8886 | 0.8862 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s500973/metrics.json` |
| acm | HAN | 88015 | 0.9022 | 0.9008 | 0.9497 | 3025 | 0 | 1.0000 | 0.8991 | 0.8975 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s88015/metrics.json` |
| acm | HAN | 251219 | 0.8965 | 0.8947 | 0.9259 | 3025 | 0 | 1.0000 | 0.8939 | 0.8919 | no | yes | `runs/class_wise_mixing/acm__han__class_wise_mixing_s251219/metrics.json` |
| acm | RGCN | 312132 | 0.9395 | 0.9386 | 0.9545 | 3025 | 0 | 1.0000 | 0.8953 | 0.8933 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s312132/metrics.json` |
| acm | RGCN | 238623 | 0.9051 | 0.9042 | 0.9038 | 3025 | 0 | 1.0000 | 0.9017 | 0.9004 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s238623/metrics.json` |
| acm | RGCN | 792965 | 0.9197 | 0.9188 | 0.9416 | 3025 | 0 | 1.0000 | 0.8843 | 0.8820 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s792965/metrics.json` |
| acm | RGCN | 15092 | 0.9179 | 0.9169 | 0.9630 | 3025 | 0 | 1.0000 | 0.8942 | 0.8928 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s15092/metrics.json` |
| acm | RGCN | 661491 | 0.9065 | 0.9060 | 0.9331 | 3025 | 0 | 1.0000 | 0.9005 | 0.8990 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s661491/metrics.json` |
| acm | RGCN | 588722 | 0.9334 | 0.9325 | 0.9410 | 3025 | 0 | 1.0000 | 0.9012 | 0.8999 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s588722/metrics.json` |
| acm | RGCN | 825661 | 0.9328 | 0.9320 | 0.9630 | 3025 | 0 | 1.0000 | 0.9065 | 0.9051 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s825661/metrics.json` |
| acm | RGCN | 500973 | 0.9169 | 0.9155 | 0.9405 | 3025 | 0 | 1.0000 | 0.8886 | 0.8862 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s500973/metrics.json` |
| acm | RGCN | 88015 | 0.9258 | 0.9249 | 0.9394 | 3025 | 0 | 1.0000 | 0.8991 | 0.8975 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s88015/metrics.json` |
| acm | RGCN | 251219 | 0.9346 | 0.9339 | 0.9620 | 3025 | 0 | 1.0000 | 0.8939 | 0.8919 | no | yes | `runs/class_wise_mixing/acm__rgcn__class_wise_mixing_s251219/metrics.json` |
| dblp | HAN | 312132 | 0.8823 | 0.8859 | 0.8838 | 4057 | 0 | 1.0000 | 0.9222 | 0.9299 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s312132/metrics.json` |
| dblp | HAN | 238623 | 0.8769 | 0.8803 | 0.8983 | 4057 | 0 | 1.0000 | 0.9175 | 0.9254 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s238623/metrics.json` |
| dblp | HAN | 792965 | 0.8732 | 0.8785 | 0.8966 | 4057 | 0 | 1.0000 | 0.9191 | 0.9271 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s792965/metrics.json` |
| dblp | HAN | 15092 | 0.8651 | 0.8690 | 0.8781 | 4057 | 0 | 1.0000 | 0.9187 | 0.9268 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s15092/metrics.json` |
| dblp | HAN | 661491 | 0.9144 | 0.9180 | 0.9177 | 4057 | 0 | 1.0000 | 0.9143 | 0.9229 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s661491/metrics.json` |
| dblp | HAN | 588722 | 0.8918 | 0.8972 | 0.8939 | 4057 | 0 | 1.0000 | 0.9294 | 0.9356 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s588722/metrics.json` |
| dblp | HAN | 825661 | 0.8848 | 0.8894 | 0.8840 | 4057 | 0 | 1.0000 | 0.9213 | 0.9289 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s825661/metrics.json` |
| dblp | HAN | 500973 | 0.8749 | 0.8799 | 0.8550 | 4057 | 0 | 1.0000 | 0.9261 | 0.9320 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s500973/metrics.json` |
| dblp | HAN | 88015 | 0.8618 | 0.8658 | 0.8664 | 4057 | 0 | 1.0000 | 0.9256 | 0.9327 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s88015/metrics.json` |
| dblp | HAN | 251219 | 0.8504 | 0.8535 | 0.8513 | 4057 | 0 | 1.0000 | 0.9198 | 0.9275 | no | yes | `runs/class_wise_mixing/dblp__han__class_wise_mixing_s251219/metrics.json` |
| dblp | RGCN | 312132 | 0.9438 | 0.9479 | 0.9556 | 4057 | 0 | 1.0000 | 0.9222 | 0.9299 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s312132/metrics.json` |
| dblp | RGCN | 238623 | 0.9309 | 0.9352 | 0.9388 | 4057 | 0 | 1.0000 | 0.9175 | 0.9254 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s238623/metrics.json` |
| dblp | RGCN | 792965 | 0.9424 | 0.9465 | 0.9444 | 4057 | 0 | 1.0000 | 0.9191 | 0.9271 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s792965/metrics.json` |
| dblp | RGCN | 15092 | 0.9386 | 0.9433 | 0.9500 | 4057 | 0 | 1.0000 | 0.9187 | 0.9268 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s15092/metrics.json` |
| dblp | RGCN | 661491 | 0.9414 | 0.9461 | 0.9556 | 4057 | 0 | 1.0000 | 0.9143 | 0.9229 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s661491/metrics.json` |
| dblp | RGCN | 588722 | 0.9344 | 0.9405 | 0.9494 | 4057 | 0 | 1.0000 | 0.9294 | 0.9356 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s588722/metrics.json` |
| dblp | RGCN | 825661 | 0.9423 | 0.9465 | 0.9433 | 4057 | 0 | 1.0000 | 0.9213 | 0.9289 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s825661/metrics.json` |
| dblp | RGCN | 500973 | 0.9337 | 0.9394 | 0.9497 | 4057 | 0 | 1.0000 | 0.9261 | 0.9320 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s500973/metrics.json` |
| dblp | RGCN | 88015 | 0.9439 | 0.9479 | 0.9499 | 4057 | 0 | 1.0000 | 0.9256 | 0.9327 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s88015/metrics.json` |
| dblp | RGCN | 251219 | 0.9349 | 0.9398 | 0.9449 | 4057 | 0 | 1.0000 | 0.9198 | 0.9275 | no | yes | `runs/class_wise_mixing/dblp__rgcn__class_wise_mixing_s251219/metrics.json` |
| imdb | HAN | 312132 | 0.4356 | 0.7434 | 0.4674 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s312132/metrics.json` |
| imdb | HAN | 238623 | 0.4365 | 0.7403 | 0.4867 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s238623/metrics.json` |
| imdb | HAN | 792965 | 0.4387 | 0.7413 | 0.4756 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s792965/metrics.json` |
| imdb | HAN | 15092 | 0.4431 | 0.7437 | 0.5012 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s15092/metrics.json` |
| imdb | HAN | 661491 | 0.4319 | 0.7422 | 0.4561 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s661491/metrics.json` |
| imdb | HAN | 588722 | 0.4375 | 0.7445 | 0.4757 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s588722/metrics.json` |
| imdb | HAN | 825661 | 0.4514 | 0.7443 | 0.5002 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s825661/metrics.json` |
| imdb | HAN | 500973 | 0.4566 | 0.7458 | 0.4836 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s500973/metrics.json` |
| imdb | HAN | 88015 | 0.5008 | 0.3788 | 0.5058 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s88015/metrics.json` |
| imdb | HAN | 251219 | 0.4355 | 0.7433 | 0.4787 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__han__class_wise_mixing_s251219/metrics.json` |
| imdb | RGCN | 312132 | 0.6429 | 0.7921 | 0.6776 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s312132/metrics.json` |
| imdb | RGCN | 238623 | 0.6319 | 0.7933 | 0.6801 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s238623/metrics.json` |
| imdb | RGCN | 792965 | 0.6290 | 0.7923 | 0.6816 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s792965/metrics.json` |
| imdb | RGCN | 15092 | 0.6279 | 0.7883 | 0.6774 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s15092/metrics.json` |
| imdb | RGCN | 661491 | 0.6308 | 0.7927 | 0.6837 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s661491/metrics.json` |
| imdb | RGCN | 588722 | 0.6340 | 0.7907 | 0.6831 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s588722/metrics.json` |
| imdb | RGCN | 825661 | 0.6363 | 0.7903 | 0.6822 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s825661/metrics.json` |
| imdb | RGCN | 500973 | 0.6348 | 0.7911 | 0.6765 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s500973/metrics.json` |
| imdb | RGCN | 88015 | 0.6370 | 0.7903 | 0.6850 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s88015/metrics.json` |
| imdb | RGCN | 251219 | 0.6362 | 0.7907 | 0.6803 | 4564 | 368 | 0.9254 | 0.0000 | 0.6542 | yes | yes | `runs/class_wise_mixing/imdb__rgcn__class_wise_mixing_s251219/metrics.json` |
| freebase | HAN | 312132 | 0.1707 | 0.6368 | 0.3624 | 676 | 5324 | 0.1127 | 0.0025 | 0.0088 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s312132/metrics.json` |
| freebase | HAN | 238623 | 0.1475 | 0.6459 | 0.3444 | 650 | 5350 | 0.1083 | 0.0013 | 0.0045 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s238623/metrics.json` |
| freebase | HAN | 792965 | 0.1863 | 0.6403 | 0.3694 | 805 | 5195 | 0.1342 | 0.0010 | 0.0036 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s792965/metrics.json` |
| freebase | HAN | 15092 | 0.0000 | nan | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | nan | yes | no | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s15092/metrics.json` |
| freebase | HAN | 661491 | 0.1380 | 0.6407 | 0.3694 | 620 | 5380 | 0.1033 | 0.0020 | 0.0071 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s661491/metrics.json` |
| freebase | HAN | 588722 | 0.1383 | 0.5657 | 0.4037 | 637 | 5363 | 0.1062 | 0.0006 | 0.0022 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s588722/metrics.json` |
| freebase | HAN | 825661 | 0.1457 | 0.6052 | 0.4200 | 648 | 5352 | 0.1080 | 0.0025 | 0.0087 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s825661/metrics.json` |
| freebase | HAN | 500973 | 0.2057 | 0.5932 | 0.4178 | 707 | 5293 | 0.1178 | 0.0006 | 0.0020 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s500973/metrics.json` |
| freebase | HAN | 88015 | 0.1340 | 0.5831 | 0.3267 | 607 | 5393 | 0.1012 | 0.0006 | 0.0023 | yes | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s88015/metrics.json` |
| freebase | HAN | 251219 | 0.1898 | 0.7043 | 0.4349 | 589 | 5411 | 0.0982 | 0.1380 | 0.7067 | no | yes | `runs/class_wise_mixing/freebase__han__class_wise_mixing_s251219/metrics.json` |
| freebase | RGCN | 312132 | 0.1728 | 0.6499 | 0.4185 | 676 | 5324 | 0.1127 | 0.0025 | 0.0088 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s312132/metrics.json` |
| freebase | RGCN | 238623 | 0.1762 | 0.6793 | 0.4206 | 650 | 5350 | 0.1083 | 0.0013 | 0.0045 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s238623/metrics.json` |
| freebase | RGCN | 792965 | 0.1920 | 0.6475 | 0.3556 | 805 | 5195 | 0.1342 | 0.0010 | 0.0036 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s792965/metrics.json` |
| freebase | RGCN | 15092 | 0.0000 | nan | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | nan | yes | no | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s15092/metrics.json` |
| freebase | RGCN | 661491 | 0.2063 | 0.6572 | 0.4226 | 620 | 5380 | 0.1033 | 0.0020 | 0.0071 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s661491/metrics.json` |
| freebase | RGCN | 588722 | 0.1849 | 0.6036 | 0.3429 | 637 | 5363 | 0.1062 | 0.0006 | 0.0022 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s588722/metrics.json` |
| freebase | RGCN | 825661 | 0.1714 | 0.6117 | 0.3295 | 648 | 5352 | 0.1080 | 0.0025 | 0.0087 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s825661/metrics.json` |
| freebase | RGCN | 500973 | 0.3477 | 0.6493 | 0.5383 | 707 | 5293 | 0.1178 | 0.0006 | 0.0020 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s500973/metrics.json` |
| freebase | RGCN | 88015 | 0.2319 | 0.5968 | 0.2790 | 607 | 5393 | 0.1012 | 0.0006 | 0.0023 | yes | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s88015/metrics.json` |
| freebase | RGCN | 251219 | 0.1768 | 0.6899 | 0.4038 | 589 | 5411 | 0.0982 | 0.1380 | 0.7067 | no | yes | `runs/class_wise_mixing/freebase__rgcn__class_wise_mixing_s251219/metrics.json` |
| mag | HAN | 312132 | 0.0141 | 0.1287 | 0.0325 | 5886 | 114 | 0.9810 | 0.0512 | 0.2632 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s312132/metrics.json` |
| mag | HAN | 238623 | 0.0100 | 0.1726 | 0.0328 | 5915 | 85 | 0.9858 | 0.0104 | 0.1726 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s238623/metrics.json` |
| mag | HAN | 792965 | 0.0317 | 0.1786 | 0.0294 | 5907 | 93 | 0.9845 | 0.0293 | 0.2302 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s792965/metrics.json` |
| mag | HAN | 15092 | 0.0109 | 0.0513 | 0.0746 | 5918 | 82 | 0.9863 | 0.0280 | 0.0769 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s15092/metrics.json` |
| mag | HAN | 661491 | 0.0183 | 0.1311 | 0.0253 | 5881 | 119 | 0.9802 | 0.0134 | 0.1844 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s661491/metrics.json` |
| mag | HAN | 588722 | 0.0344 | 0.1242 | 0.0281 | 5887 | 113 | 0.9812 | 0.0153 | 0.1438 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s588722/metrics.json` |
| mag | HAN | 825661 | 0.0142 | 0.1624 | 0.0303 | 5914 | 86 | 0.9857 | 0.0092 | 0.1624 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s825661/metrics.json` |
| mag | HAN | 500973 | 0.0259 | 0.2857 | 0.0344 | 5922 | 78 | 0.9870 | 0.0355 | 0.3233 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s500973/metrics.json` |
| mag | HAN | 88015 | 0.0174 | 0.1316 | 0.0321 | 5895 | 105 | 0.9825 | 0.0050 | 0.1513 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s88015/metrics.json` |
| mag | HAN | 251219 | 0.0087 | 0.0538 | 0.0170 | 5912 | 88 | 0.9853 | 0.0119 | 0.0860 | no | yes | `runs/class_wise_mixing/mag__han__class_wise_mixing_s251219/metrics.json` |
| mag | RGCN | 312132 | 0.1027 | 0.2924 | 0.0989 | 5886 | 114 | 0.9810 | 0.0512 | 0.2632 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s312132/metrics.json` |
| mag | RGCN | 238623 | 0.0324 | 0.2965 | 0.1030 | 5915 | 85 | 0.9858 | 0.0104 | 0.1726 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s238623/metrics.json` |
| mag | RGCN | 792965 | 0.0784 | 0.4444 | 0.0815 | 5907 | 93 | 0.9845 | 0.0293 | 0.2302 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s792965/metrics.json` |
| mag | RGCN | 15092 | 0.2053 | 0.2308 | 0.1329 | 5918 | 82 | 0.9863 | 0.0280 | 0.0769 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s15092/metrics.json` |
| mag | RGCN | 661491 | 0.0642 | 0.2459 | 0.0685 | 5881 | 119 | 0.9802 | 0.0134 | 0.1844 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s661491/metrics.json` |
| mag | RGCN | 588722 | 0.0876 | 0.1961 | 0.0860 | 5887 | 113 | 0.9812 | 0.0153 | 0.1438 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s588722/metrics.json` |
| mag | RGCN | 825661 | 0.0735 | 0.4017 | 0.0969 | 5914 | 86 | 0.9857 | 0.0092 | 0.1624 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s825661/metrics.json` |
| mag | RGCN | 500973 | 0.1113 | 0.3910 | 0.0721 | 5922 | 78 | 0.9870 | 0.0355 | 0.3233 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s500973/metrics.json` |
| mag | RGCN | 88015 | 0.0870 | 0.1579 | 0.0797 | 5895 | 105 | 0.9825 | 0.0050 | 0.1513 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s88015/metrics.json` |
| mag | RGCN | 251219 | 0.0382 | 0.1720 | 0.0632 | 5912 | 88 | 0.9853 | 0.0119 | 0.0860 | no | yes | `runs/class_wise_mixing/mag__rgcn__class_wise_mixing_s251219/metrics.json` |

## Diagnostics / Caveats

- Freebase GTN attention diagnostic: 18/20 completed Freebase runs contain NaN values in `gtn_attentions`; 18/20 have finite final `test_macro_f1`, `test_accuracy`, and `val_macro_f1`.
- Freebase mixed ratio: 0.0990+/-0.0353 across completed Freebase runs.
- NaN values in saved GTN attentions suggest GTN-stage instability or degenerate attention on Freebase. These runs are retained when final metrics are finite, but the attention diagnostics should not be hidden.
- Class-wise mixing can be weak when same-class same-split groups are small; singleton groups remain unchanged and are reported through `unchanged_nodes` and mixed ratio.
- MAG has many classes and uses subsampled nodes, so macro-F1 can be very low. Interpret MAG primarily through within-dataset deltas rather than absolute macro-F1.
- If this document reports fewer than 140 completed runs, the table is interim and should be regenerated after resume completes.

## Reproduction Commands

Resume missing runs:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate tda
cd /workspace/TDA
bash experiments/run_class_wise_mixing.sh
```

Count completed official runs:

```bash
find runs/class_wise_mixing -name metrics.json | grep -Ev '_s0/metrics.json$' | wc -l
```

Check missing official runs:

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

Regenerate this summary:

```bash
python experiments/summarize_class_wise_mixing.py
```

