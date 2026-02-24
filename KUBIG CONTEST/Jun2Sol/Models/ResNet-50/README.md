# FER: GCN 랜드마크 인코더 + FiLM 컨디셔닝

GCN으로 인코딩한 얼굴 랜드마크를 FiLM으로 주입해 표정 인식 성능을 높인 ResNet-50 기반 모델입니다.

**최고 성능 (GCN_FiLM_L34): 테스트 정확도 87.12%**

---

## 방법론

MediaPipe로 추출한 478개의 얼굴 랜드마크를 GCN으로 인코딩하고, FiLM을 통해 ResNet-50의 layer3와 layer4에 주입합니다.

```
입력 이미지 ──► ResNet-50 (VGGFace2 사전학습)
                    ▲ FiLM (layer3, layer4)
랜드마크    ──► GCN 인코더 ──► 256차원 임베딩
```

- **GCN**: 478개 랜드마크 → k-NN(k=8) 그래프 → 3층 GCN → 256차원
- **FiLM**: `γ(c) · x + β(c)` 형태로 CNN 피처맵에 스케일/시프트 적용

---

## 실험 결과

| 모델 | 테스트 정확도 |
|---|---|
| Baseline (ResNet-50 + VGGFace2) | 86.18% |
| **GCN_FiLM_L34 (최고 성능)** | **87.12%** |
| GCN_FiLM_L4 | 86.72% |
| GCN_FiLM_L234 | 86.91% |
| GAT_FiLM_L34 | 86.54% |

#### 클래스별 정확도 (GCN_FiLM_L34)

| 클래스 | 정확도 | 샘플 수 |
|---|---|---|
| happy | 95.4% | 1,206 |
| surprise | 88.0% | 485 |
| neutral | 87.9% | 900 |
| sad | 86.6% | 821 |
| angry | 83.4% | 428 |
| fear | 58.5% | 123 |
| disgust | 54.2% | 168 |

> disgust/fear는 클래스 불균형으로 성능이 낮음. 클래스 가중치 손실 적용 시 disgust +7.7%, fear +1.7% 향상.

---

## 사용법

### 환경 설정

```bash
pip install -r requirements.txt
```

VGGFace2 사전학습 가중치를 다운로드해 `pretrained/resnet50_ft_weight.pkl` 에 저장.

### 학습

```bash
# 최고 성능 모델 (GCN_FiLM_L34)
python train_fer_gcn_film.py --data-root ./cleaned_7class --gpus 0,1 --experiments GCN_FiLM_L34

# 클래스 가중치 학습 (소수 클래스 성능 개선)
python train_fer_cls_weight.py --data-root ./cleaned_7class --gpus 0,1
```

### 추론

```bash
# 단일 이미지
python inference.py --checkpoint ./results/best_GCN_FiLM_L34.pt --image face.jpg

# Grad-CAM 시각화 포함
python inference.py --checkpoint ./results/best_GCN_FiLM_L34.pt --image face.jpg --gradcam --save-dir ./output/
```
