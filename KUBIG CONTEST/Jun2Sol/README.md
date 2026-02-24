# 📚 Face2Emo: <br> 안면 특징 추출을 통한 감정 인식 최적화 및 맞춤형 색상 추천

# 1. Team

22기 박준영 | 23기 서민솔, 정준호

# 2. Introduction

# 3. Experiments

## 3-1. Datasets

학습에 사용한 7클래스 FER 데이터셋입니다 (RAF-DB 기반, 정제 후 33,000+ 이미지).

**다운로드**: [Google Drive](https://drive.google.com/file/d/1iO3nstMqRdVtq41dR4N_ta4lS2L7B5A8/view?usp=sharing)

| 클래스 | 샘플 수 (전처리 후) |
|---|---|
| happy | 11,926 |
| neutral | 9,245 |
| sad | 8,241 |
| angry | 4,089 |
| surprise | 4,776 |
| fear | 1,310 |
| disgust | 1,725 |
| **합계** | **41,312** |

### 전처리

```bash
# 1. 얼굴 크롭
python datasets/crop_faces.py --src ./raw_data --dst ./cropped_7class

# 2. 저품질 이미지 정제 (블러, 중복, 랜드마크 미검출 제거)
python datasets/clean_dataset.py --src ./cropped_7class --dst ./cleaned_7class
```

## 3-2. Models

## 3-3. Color matching

## 3-4. User Interface

# 4. Results
