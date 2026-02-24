"""
감정 기반 컬러 추천 시스템
Stream A: FER (EfficientNet-B2, AffectNet pretrained)
Stream B: MediaPipe 랜드마크 기반 얼굴 색상 추출
Stream C: 동적 색상 보정 (16 그리드 × LAB 조정)
"""
import warnings, base64, io
warnings.filterwarnings('ignore')

import numpy as np
from skimage import color as skcolor
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ════════════════════════════════════════════════════════════
# 1. 정의
# ════════════════════════════════════════════════════════════
CLASSES    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_KR = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포',
    'happy': '행복', 'neutral': '중립', 'sad': '슬픔', 'surprise': '놀람'
}
EMOTION_EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# ── Stream C 감정별 기본 팔레트 (단일 기준색) ─────────────────
# 감정의 반대 방향으로 보완하는 색상
BASE_COLOR = {
    'angry':    '#90E0C3',  # 쿨 민트   → 진정/냉각
    'sad':      '#FFB3A7',  # 소프트코랄 → 활력부여
    'fear':     '#D8C3A5',  # 웜 샌드   → 안정감/온기
    'surprise': '#1A237E',  # 딥 네이비  → 평정심/차분
    'disgust':  '#DCD0FF',  # 클린 라벤더 → 정화/쾌적
    'happy':    '#FFFACD',  # 크림 아이보리 → 긍정 확장
    'neutral':  '#36454F',  # 차콜       → 안전/전문성
}
COLOR_NAME = {
    'angry':    '쿨 민트',
    'sad':      '소프트 코랄',
    'fear':     '웜 샌드',
    'surprise': '딥 네이비',
    'disgust':  '클린 라벤더',
    'happy':    '크림 아이보리',
    'neutral':  '차콜',
}
COLOR_DESC = {
    'angry':    '붉어진 피부 톤을 중화하고 시각적으로 시원한 느낌',
    'sad':      '부드러운 오렌지빛으로 따뜻한 에너지를 불어넣음',
    'fear':     '대지의 차분한 베이지로 심리적 지지감 제공',
    'surprise': '깊고 진중한 파란색으로 들뜬 신경을 가라앉힘',
    'disgust':  '깨끗한 라벤더로 불쾌한 감정을 씻어냄',
    'happy':    '크리미한 톤으로 행복감을 편안하게 유지',
    'neutral':  '시각적 자극 최소화, 신뢰감과 세련미 전달',
}

# 컨디션 점수 가중치
COND_WEIGHTS = {
    'happy': 50, 'surprise': 20, 'neutral': 0,
    'sad': -20,  'fear': -30,    'angry': -35, 'disgust': -40
}

# 감정 바 색상 (UI)
BAR_COLOR = {
    'angry':'#DC143C', 'disgust':'#808000', 'fear':'#7B68EE',
    'happy':'#FFD700', 'neutral':'#C4A882', 'sad':'#4169E1', 'surprise':'#FF8C00'
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ════════════════════════════════════════════════════════════
# 2. 모델 로드
# ════════════════════════════════════════════════════════════
def load_model():
    sd = torch.load('/home/elicer/best_fer_affectnet.pth',
                    map_location='cpu', weights_only=True)
    new_sd = {}
    for k, v in sd.items():
        if   k == 'classifier.1.weight': new_sd['classifier.weight'] = v
        elif k == 'classifier.1.bias':   new_sd['classifier.bias']   = v
        else:                            new_sd[k] = v
    m = timm.create_model('tf_efficientnet_b2', num_classes=7, pretrained=False)
    m.load_state_dict(new_sd, strict=True)
    return m.eval().to(DEVICE)

model = load_model()
print(f'FER model loaded on {DEVICE}')

# ════════════════════════════════════════════════════════════
# 3-0. MediaPipe FaceLandmarker 초기화
# ════════════════════════════════════════════════════════════
FACE_MODEL_PATH = '/home/elicer/cleaned_7class/cleaned_7class/face_landmarker_v2_with_blendshapes.task'

# 볼(피부) 랜드마크 인덱스 (양쪽)
CHEEK_IDX = [116,117,118,119,120,121,100,142,203,206,
             345,346,347,348,349,350,329,371,423,426]
# 홍채(눈동자) 랜드마크 인덱스 (양쪽, 478개 중 468~477)
IRIS_IDX  = [468,469,470,471,472, 473,474,475,476,477]

def init_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

landmarker = init_landmarker()
print('FaceLandmarker loaded')

# ════════════════════════════════════════════════════════════
# 3. Stream B — 얼굴 색상 추출
# ════════════════════════════════════════════════════════════
def dominant_color(pixels: np.ndarray, k: int = 3) -> np.ndarray:
    """K-means로 주요 RGB 색상 추출 (float [0,1] 반환), 최대 2000px 서브샘플"""
    if len(pixels) < k:
        return np.array([0.5, 0.5, 0.5])
    if len(pixels) > 2000:
        idx = np.random.default_rng(42).choice(len(pixels), 2000, replace=False)
        pixels = pixels[idx]
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=50, algorithm='lloyd')
    km.fit(pixels)
    counts = np.bincount(km.labels_)
    return km.cluster_centers_[counts.argmax()]

def _sample_lm_pixels(img_rgb, indices, landmarks, h, w, radius=10):
    """랜드마크 인덱스 주변 패치 픽셀 수집 → float [0,1]"""
    patches = []
    for i in indices:
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        cx, cy = int(lm.x * w), int(lm.y * h)
        y1, y2 = max(0, cy-radius), min(h, cy+radius)
        x1, x2 = max(0, cx-radius), min(w, cx+radius)
        patch = img_rgb[y1:y2, x1:x2].reshape(-1, 3)
        if len(patch) > 0:
            patches.append(patch)
    if not patches:
        return None
    return np.vstack(patches).astype(float) / 255.0

def draw_face_landmarks(img_rgb: np.ndarray, landmarks) -> np.ndarray:
    """랜드마크 점+주요 윤곽선을 이미지에 그려 반환 (RGB ndarray)"""
    import cv2
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()

    # 주요 윤곽 연결선 (얼굴 외곽, 눈, 눈썹, 코, 입술)
    CONTOUR_PAIRS = [
        # 얼굴 외곽 (silhouette)
        10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
        152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,
    ]
    # 눈 윤곽
    L_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
    R_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
    # 눈썹
    L_BROW = [70,63,105,66,107,55,65,52,53,46]
    R_BROW = [300,293,334,296,336,285,295,282,283,276]
    # 코
    NOSE   = [168,6,197,195,5,4,1,19,94,2,164,0,11,302,303,271,304,272,310,311,312,13,82,81,80,40,39,37,0]
    # 입술 외곽
    LIPS   = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]

    dot_color  = (0, 230, 180)   # 청록
    line_color = (80, 220, 160)  # 연한 청록
    iris_color = (255, 200, 50)  # 황금

    # 모든 랜드마크 점 (작게)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (cx, cy), 1, dot_color, -1, cv2.LINE_AA)

    # 윤곽 연결선 그리기 헬퍼
    def draw_path(pts, closed=False, color=line_color, thickness=1):
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x*w), int(landmarks[a].y*h))
                p2 = (int(landmarks[b].x*w), int(landmarks[b].y*h))
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
        if closed and len(pts) >= 2:
            a, b = pts[-1], pts[0]
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x*w), int(landmarks[a].y*h))
                p2 = (int(landmarks[b].x*w), int(landmarks[b].y*h))
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

    draw_path(L_EYE,  closed=True)
    draw_path(R_EYE,  closed=True)
    draw_path(L_BROW)
    draw_path(R_BROW)
    draw_path(NOSE)
    draw_path(LIPS,   closed=True)

    # 홍채 원
    for iris_set in [IRIS_IDX[:5], IRIS_IDX[5:]]:
        pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h))
               for i in iris_set if i < len(landmarks)]
        if len(pts) >= 3:
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            r  = int(np.mean([np.hypot(p[0]-cx, p[1]-cy) for p in pts])) + 2
            cv2.circle(out, (cx, cy), r, iris_color, 1, cv2.LINE_AA)

    # 얼굴 영역 점선 박스
    face_xs_px = [int(lm.x * w) for lm in landmarks]
    face_ys_px = [int(lm.y * h) for lm in landmarks]
    pad = 14
    bx1 = max(0,   min(face_xs_px) - pad)
    by1 = max(0,   min(face_ys_px) - pad)
    bx2 = min(w-1, max(face_xs_px) + pad)
    by2 = min(h-1, max(face_ys_px) + pad)
    dash, gap, rect_col = 10, 6, (0, 210, 255)
    for x in range(bx1, bx2, dash+gap):
        cv2.line(out, (x, by1), (min(x+dash, bx2), by1), rect_col, 2, cv2.LINE_AA)
        cv2.line(out, (x, by2), (min(x+dash, bx2), by2), rect_col, 2, cv2.LINE_AA)
    for y in range(by1, by2, dash+gap):
        cv2.line(out, (bx1, y), (bx1, min(y+dash, by2)), rect_col, 2, cv2.LINE_AA)
        cv2.line(out, (bx2, y), (bx2, min(y+dash, by2)), rect_col, 2, cv2.LINE_AA)

    return out


def extract_face_colors(img_rgb: np.ndarray):
    """
    MediaPipe FaceLandmarker로 랜드마크 검출 후 색상 추출.
    얼굴 미검출 시 영역 기반 fallback.
    Returns: skin_rgb, hair_rgb, eye_rgb (float [0,1]), landmarks_or_None
    """
    h, w = img_rgb.shape[:2]

    # ── MediaPipe 랜드마크 검출 ────────────────────────────
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_img)

    if result.face_landmarks:
        lms = result.face_landmarks[0]

        # 피부: 양쪽 볼 랜드마크
        skin_px = _sample_lm_pixels(img_rgb, CHEEK_IDX, lms, h, w, radius=12)
        skin_rgb = dominant_color(skin_px) if skin_px is not None and len(skin_px)>=3 \
                   else np.array([0.8, 0.6, 0.5])

        # 눈동자: 홍채 랜드마크 → 가장 어두운 클러스터
        eye_px = _sample_lm_pixels(img_rgb, IRIS_IDX, lms, h, w, radius=6)
        if eye_px is not None and len(eye_px) >= 4:
            k4 = min(4, len(eye_px))
            km = KMeans(n_clusters=k4, n_init=3, random_state=42, max_iter=50, algorithm='lloyd')
            km.fit(eye_px)
            eye_rgb = km.cluster_centers_[km.cluster_centers_.mean(axis=1).argsort()[0]]
        else:
            eye_rgb = np.array([0.25, 0.20, 0.15])

        # 머리카락: face bounding box 기반 + 피부색/밝은배경 제외
        face_xs   = [lm.x for lm in lms]
        face_ys   = [lm.y for lm in lms]
        fx1       = max(0,   int(min(face_xs) * w) - int(w*0.05))
        fx2       = min(w,   int(max(face_xs) * w) + int(w*0.05))
        face_top_y = int(min(face_ys) * h)
        face_h_px  = int((max(face_ys) - min(face_ys)) * h)
        hair_top  = max(0, face_top_y - int(face_h_px * 0.55))
        hair_bot  = max(1, face_top_y + int(face_h_px * 0.06))
        if hair_bot > hair_top and fx2 > fx1:
            hair_region = img_rgb[hair_top:hair_bot, fx1:fx2].astype(float) / 255.0
            hair_px     = hair_region.reshape(-1, 3)
            if len(hair_px) >= 3:
                skin_dist  = np.linalg.norm(hair_px - skin_rgb, axis=1)
                brightness = hair_px.mean(axis=1)
                mask       = (skin_dist > 0.18) & (brightness < 0.88)
                filtered   = hair_px[mask]
                hair_rgb   = dominant_color(filtered if len(filtered) >= 3 else hair_px)
            else:
                hair_rgb = np.array([0.2, 0.15, 0.1])
        else:
            hair_rgb = np.array([0.2, 0.15, 0.1])

        return skin_rgb, hair_rgb, eye_rgb, lms

    else:
        # ── Fallback: 영역 기반 ───────────────────────────
        print('[WARN] 얼굴 미검출 → 영역 기반 fallback')
        img_f = img_rgb.astype(float) / 255.0
        skin_px = img_f[int(h*.50):int(h*.75), int(w*.20):int(w*.80)].reshape(-1, 3)
        skin_rgb = dominant_color(skin_px)
        hair_px  = img_f[0:max(1,int(h*.18)), int(w*.10):int(w*.90)].reshape(-1, 3)
        hair_rgb = dominant_color(hair_px)
        eye_px   = img_f[int(h*.28):int(h*.42), int(w*.15):int(w*.85)].reshape(-1, 3)
        if len(eye_px) >= 4:
            km = KMeans(n_clusters=4, n_init=3, random_state=42, max_iter=50, algorithm='lloyd')
            km.fit(eye_px)
            eye_rgb = km.cluster_centers_[km.cluster_centers_.mean(axis=1).argsort()[0]]
        else:
            eye_rgb = np.array([0.25, 0.20, 0.15])

        return skin_rgb, hair_rgb, eye_rgb, None

# ════════════════════════════════════════════════════════════
# 4. Stream B — LAB 변환 & 16 그리드
# ════════════════════════════════════════════════════════════
def rgb_to_lab(rgb_float: np.ndarray) -> np.ndarray:
    """float [0,1] RGB → LAB (L:0~100, A/B:-128~127)"""
    return skcolor.rgb2lab(rgb_float.reshape(1, 1, 3))[0, 0]

def hex_to_rgb_float(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return np.array([r, g, b], dtype=float) / 255.0

def rgb_float_to_hex(rgb_f: np.ndarray) -> str:
    rgb = np.clip(rgb_f * 255, 0, 255).astype(int)
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def get_16grid(skin_lab, hair_lab, eye_lab):
    """
    X축 (온도): 피부 LAB-B 단독
      temp_score = (skin_B + 128) / 255 * 100  (B ∈ [-128,127] → [0,100])
    Y축 (명도): 피부:머리:눈 = 7:2:1 가중 평균 L값
      light_score = (skin_L*0.7 + hair_L*0.2 + eye_L*0.1) / 100 * 100
    Returns:
      temp_score  : 0~100
      light_score : 0~100
      temp_label  : str
      light_label : str
      temp_adj    : LAB-B 보정량
      light_adj   : LAB-L 보정량
    """
    # 온도 점수
    skin_B = float(skin_lab[2])
    temp_score = (skin_B + 128) / 255.0 * 100

    # 명도 점수
    skin_L  = float(skin_lab[0])
    hair_L  = float(hair_lab[0])
    eye_L   = float(eye_lab[0])
    light_score = (skin_L*0.7 + hair_L*0.2 + eye_L*0.1)  # L ∈ [0,100]

    # 온도 구간
    if   temp_score < 25:  temp_label, temp_adj = 'True Cool',  -20
    elif temp_score < 50:  temp_label, temp_adj = 'Soft Cool',  -10
    elif temp_score < 75:  temp_label, temp_adj = 'Soft Warm',  +10
    else:                  temp_label, temp_adj = 'True Warm',  +20

    # 명도 구간
    if   light_score < 25:  light_label, light_adj = 'Very Dark',   -20
    elif light_score < 50:  light_label, light_adj = 'Medium Dark', -10
    elif light_score < 75:  light_label, light_adj = 'Medium Light',+10
    else:                   light_label, light_adj = 'Very Light',  +20

    return {
        'temp_score':  round(temp_score, 1),
        'light_score': round(light_score, 1),
        'temp_label':  temp_label,
        'light_label': light_label,
        'temp_adj':    temp_adj,
        'light_adj':   light_adj,
    }

# ════════════════════════════════════════════════════════════
# 5. Stream C — 동적 색상 보정
# ════════════════════════════════════════════════════════════
def adjust_color(base_hex: str, light_adj: float, temp_adj: float) -> str:
    """
    기본 감정 색상 → LAB 변환 → L+light_adj, B+temp_adj → RGB 변환
    """
    rgb_f  = hex_to_rgb_float(base_hex)
    lab    = rgb_to_lab(rgb_f).copy()
    lab[0] = np.clip(lab[0] + light_adj, 0,   100)   # L
    lab[2] = np.clip(lab[2] + temp_adj,  -128, 127)   # B
    adj_rgb = skcolor.lab2rgb(lab.reshape(1, 1, 3))[0, 0]
    return rgb_float_to_hex(adj_rgb)

# ════════════════════════════════════════════════════════════
# 6. Flask API
# ════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file    = request.files['image']
    img_pil = Image.open(file.stream).convert('RGB')

    # K-means 속도 최적화: 색상 추출용 이미지를 최대 512px로 제한
    w, h = img_pil.size
    if max(w, h) > 512:
        scale = 512 / max(w, h)
        img_small = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    else:
        img_small = img_pil
    img_rgb = np.array(img_small)

    # ── Stream A: 감정 추론 ─────────────────────────────────
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs_t = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    probs     = {e: float(p) for e, p in zip(CLASSES, probs_t)}
    top       = max(probs, key=probs.get)

    # 컨디션 점수
    raw   = sum(probs[e] * COND_WEIGHTS[e] for e in CLASSES)
    score = float(np.clip((raw + 40) / 90 * 100, 0, 100))

    # ── Stream B: 얼굴 색상 추출 → LAB ─────────────────────
    skin_f, hair_f, eye_f, landmarks = extract_face_colors(img_rgb)
    skin_lab = rgb_to_lab(skin_f)
    hair_lab = rgb_to_lab(hair_f)
    eye_lab  = rgb_to_lab(eye_f)
    skin_hex = rgb_float_to_hex(skin_f)
    hair_hex = rgb_float_to_hex(hair_f)
    eye_hex  = rgb_float_to_hex(eye_f)

    # ── Stream B: 16 그리드 계산 ────────────────────────────
    grid = get_16grid(skin_lab, hair_lab, eye_lab)

    # ── Stream C: 동적 색상 보정 ────────────────────────────
    base_hex     = BASE_COLOR[top]
    adjusted_hex = adjust_color(base_hex, grid['light_adj'], grid['temp_adj'])

    # 원본 이미지 base64
    buf_orig = io.BytesIO()
    img_small.save(buf_orig, format='JPEG', quality=85)
    orig_b64 = base64.b64encode(buf_orig.getvalue()).decode()

    # 랜드마크 시각화 base64
    if landmarks is not None:
        lm_pil = Image.fromarray(draw_face_landmarks(img_rgb, landmarks))
    else:
        lm_pil = img_small
    buf_lm = io.BytesIO()
    lm_pil.save(buf_lm, format='JPEG', quality=85)
    lm_b64 = base64.b64encode(buf_lm.getvalue()).decode()

    return jsonify({
        # 이미지
        'image_b64':    orig_b64,
        'lm_b64':       lm_b64,
        # 감정
        'probs':        probs,
        'bar_colors':   BAR_COLOR,
        'top_emotion':  top,
        'top_kr':       EMOTION_KR[top],
        'top_emoji':    EMOTION_EMOJI[top],
        # 컨디션
        'score':        round(score, 1),
        # 얼굴 색상
        'skin_hex':     skin_hex,
        'hair_hex':     hair_hex,
        'eye_hex':      eye_hex,
        'skin_rgb':     [int(x*255) for x in skin_f],
        'hair_rgb':     [int(x*255) for x in hair_f],
        'eye_rgb':      [int(x*255) for x in eye_f],
        # 16 그리드
        'grid':         grid,
        # 색상 추천
        'base_hex':     base_hex,
        'adjusted_hex': adjusted_hex,
        'color_name':   COLOR_NAME[top],
        'color_desc':   COLOR_DESC[top],
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
