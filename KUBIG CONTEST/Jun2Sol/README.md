# 📚 Face2Emo: <br> 안면 특징 추출을 통한 감정 인식 최적화 및 맞춤형 색상 추천

# 1. Team

22기 박준영 | 23기 서민솔, 정준호

# 2. Introduction

# 3. Experiments

## 3-1. Datasets

학습에 사용한 7클래스 FER 데이터셋입니다 (RAF-DB 기반).

**다운로드**: [Google Drive](https://drive.google.com/file/d/1iO3nstMqRdVtq41dR4N_ta4lS2L7B5A8/view?usp=sharing)

### 전처리

```bash
# 1. 얼굴 크롭
python datasets/crop_faces.py --src ./raw_data --dst ./cropped_7class

# 2. 저품질 이미지 정제 (블러, 중복, 랜드마크 미검출 제거)
python datasets/clean_dataset.py --src ./cropped_7class --dst ./cleaned_7class
```

| 클래스 | happy | neutral | sad | surprise | angry | disgust | fear | 합계 |
|---|---|---|---|---|---|---|---|---|
| 샘플 수 | 11,926 | 9,245 | 8,241 | 4,776 | 4,089 | 1,725 | 1,310 | **41,312** |

## 3-2. Models

### ResNet-50 + GCN_FiLM

VGGFace2로 사전학습된 ResNet-50에 GCN 기반 랜드마크 정보를 FiLM으로 주입한 모델입니다.

**VGGFace2 사전학습 가중치**

[Google Drive](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU)에서 `resnet50_ft_weight.pkl` 다운로드 후 아래 경로에 저장:
```
pretrained/resnet50_ft_weight.pkl
```

**구조**

```
입력 이미지 ──► ResNet-50 (VGGFace2 사전학습)
                    ▲ FiLM (layer3, layer4)
랜드마크    ──► GCN 인코더 ──► 256차원 임베딩
```

- **랜드마크**: MediaPipe로 478개 얼굴 키포인트 추출
- **GCN**: k-NN(k=8) 그래프 위에서 3층 GCN → 256차원 임베딩
- **FiLM**: `γ(c)·x + β(c)` 형태로 ResNet feature map에 스케일/시프트 적용
- **주입 위치**: layer3 + layer4 (ablation 결과 최적)

**학습**

```bash
# 최고 성능 모델 (GCN_FiLM_L34)
python Models/ResNet-50/train_fer_gcn_film.py --data-root ./cleaned_7class --gpus 0,1 --experiments GCN_FiLM_L34

# 클래스 가중치 학습 (소수 클래스 성능 개선)
python Models/ResNet-50/train_fer_cls_weight.py --data-root ./cleaned_7class --gpus 0,1
```

**추론**

```bash
python Models/ResNet-50/inference.py --checkpoint ./results/best_GCN_FiLM_L34.pt --image face.jpg
```


## 3-3. Color matching

## 3-4. User Interface

# 4. Results
