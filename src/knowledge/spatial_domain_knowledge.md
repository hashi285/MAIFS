# 공간 분석 도메인 지식
**Spatial Consistency Analysis for Image Manipulation Detection**

---

## 📚 과학적 근거

### 핵심 논문
1. **"OmniGuard: Hybrid Manipulation Localization via Augmented Versatile Deep Image Watermarking"**
   - OmniGuard 프로젝트에서 제공되는 ViT 기반 조작 영역 탐지
   - 픽셀 수준 조작 마스크 생성

2. **"ManTra-Net: Manipulation Tracing Network" (2019)**
   - 다양한 조작 유형 탐지
   - 복사-붙여넣기, 스플라이싱, 리터칭

3. **"TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization" (CVPR 2023)**
   - RGB + Noiseprint++ 기반 멀티모달 분석
   - 조작 영역 픽셀 수준 탐지

---

## 🔬 분석 원리

### 1. 공간 불일치란?
```
자연 이미지: 모든 부분이 동일한 카메라/조명/과정으로 생성
조작 이미지: 서로 다른 출처의 영역 합성 → 불일치 발생
```

**탐지 대상**:
1. **복사-붙여넣기**: 같은 이미지 내 다른 영역 복사
2. **스플라이싱**: 다른 이미지에서 영역 붙여넣기
3. **인페인팅**: AI로 일부 영역 채움

### 2. ViT (Vision Transformer) 분석
```
이미지 → 패치 분할 (16x16) → Self-Attention → 불일치 영역
```

**Self-Attention의 장점**:
- 패치 간 관계 학습
- 불일치 영역 자동 발견
- 조작 경계(boundary) 탐지

**학습 과정**:
- 정상 이미지: 패치 간 일관성 높음
- 조작 이미지: 특정 패치만 다른 패턴
- ViT가 이 차이를 학습

---

## 📊 메트릭 해석 가이드

### 1. spatial_consistency (공간 일관성)
**측정 방법**: ViT의 attention map 분석

| 범위 | 의미 | 설명 |
|------|------|------|
| 0.9 - 1.0 | 매우 일관적 | 모든 영역이 동일 출처. 자연 이미지 |
| 0.7 - 0.9 | 대체로 일관적 | 약간의 후처리 가능. 원본일 가능성 높음 |
| 0.5 - 0.7 | 불일치 존재 | 조작 의심. 특정 영역 검사 필요 |
| < 0.5 | 심각한 불일치 | 조작 확실 또는 합성 이미지 |

### 2. manipulation_regions (조작 영역)
**출력 형식**:
```python
[
    {"x": 120, "y": 80, "width": 200, "height": 150, "confidence": 0.87},
    {"x": 400, "y": 300, "width": 100, "height": 100, "confidence": 0.72}
]
```

**해석**:
- 조작 의심 영역의 바운딩 박스
- confidence: 해당 영역이 조작됐을 확률

### 3. texture_inconsistency (텍스처 불일치)
**측정 방법**: 지역 텍스처 패턴 비교

| 값 | 의미 | 원인 |
|----|------|------|
| HIGH | 텍스처 급격히 변화 | 다른 이미지 붙여넣기 |
| MEDIUM | 약한 불일치 | AI 인페인팅, 블렌딩 |
| LOW | 텍스처 일관적 | 자연 이미지 |

---

## ⚖️ 분석의 강점과 한계

### 강점
✅ **조작 위치 특정**
   - 픽셀 수준 조작 영역 탐지
   - 다른 분석은 "전체 판정"만 제공

✅ **다양한 조작 유형 탐지**
   - 복사-붙여넣기
   - 스플라이싱
   - AI 인페인팅
   - 객체 제거/추가

✅ **시각적 증거 제공**
   - 조작 영역 하이라이트
   - 사람이 확인 가능

### 한계
❌ **완벽한 블렌딩 탐지 어려움**
   - 전문가 수준의 포토샵 작업
   - AI 기반 seamless cloning

❌ **전역 생성 이미지는 판단 어려움**
   - GAN으로 전체 생성 → 모든 영역이 동일 출처
   - "불일치"가 없음

❌ **계산 비용**
   - ViT는 계산 집약적
   - 고해상도 이미지는 느림

---

## 🤝 다른 분석과의 관계

### vs Frequency Analysis
**보완 관계**:
```
Frequency: 전역 분석 (이미지 전체의 주파수 패턴)
Spatial: 지역 분석 (특정 영역의 불일치)
```

**함께 사용**:
```
Frequency: AI_GENERATED (격자 패턴)
Spatial: manipulation_regions = [(x, y, w, h)]

→ 해석: 실제 사진에 AI로 생성한 객체를 추가함
→ Spatial이 추가된 객체의 위치 특정
```

### vs Noise Analysis (PRNU)
**보완 관계**:
```
Noise: 전역 PRNU 일관성 측정
Spatial: 지역별 PRNU 비교
```

**조작 탐지 시**:
```
1. Noise: "이미지에 불일치 있음" (전체 평가)
2. Spatial: "이 영역이 다른 출처" (위치 특정)
```

### vs Watermark Detection
**독립적**:
- Watermark: AI 모델 식별
- Spatial: 조작 위치 찾기

**혼합 이미지 분석**:
```
실제 사진 + AI 생성 객체:
- Watermark: 부분 탐지 (신뢰도 낮음)
- Spatial: 객체 영역 특정
```

---

## 💡 해석 예시

### Case 1: 복사-붙여넣기 조작
```
spatial_consistency: 0.42
manipulation_regions: [{"x": 200, "y": 150, "width": 180, "height": 200, "confidence": 0.91}]
texture_inconsistency: HIGH
```

**해석**:
"공간 일관성이 낮으며(0.42), 이미지의 (200, 150) 위치에
180x200 크기의 조작 영역이 91% 신뢰도로 탐지되었습니다.
이 영역은 주변과 텍스처 패턴이 급격히 다르며,
복사-붙여넣기 또는 스플라이싱 조작이 의심됩니다.
ViT의 attention map에서 해당 영역이 다른 패치들과
다른 특성을 보입니다."

### Case 2: AI 인페인팅 (객체 제거)
```
spatial_consistency: 0.68
manipulation_regions: [{"x": 450, "y": 320, "width": 120, "height": 150, "confidence": 0.73}]
texture_inconsistency: MEDIUM
```

**해석**:
"중간 수준의 공간 불일치(0.68)와 함께,
(450, 320) 영역에서 조작 의심 패턴이 탐지되었습니다.
텍스처 불일치가 중간 수준인 것은 AI 인페인팅의 특징입니다.
AI 기반 객체 제거 도구(Content-Aware Fill, Stable Diffusion Inpainting)가
사용되었을 가능성이 높습니다. AI는 주변과 조화롭게 채우지만,
미세한 패턴 차이가 ViT에 의해 탐지됩니다."

### Case 3: 자연 이미지 (조작 없음)
```
spatial_consistency: 0.94
manipulation_regions: []
texture_inconsistency: LOW
```

**해석**:
"매우 높은 공간 일관성(0.94)을 보이며,
조작 의심 영역이 탐지되지 않았습니다.
이미지의 모든 영역이 동일한 텍스처 패턴과 통계적 특성을 공유하며,
이는 단일 출처(한 번의 촬영)로 생성된 이미지의 전형적인 특징입니다.
조작되지 않은 원본 이미지일 가능성이 매우 높습니다."

### Case 4: 전역 AI 생성 (판단 어려움)
```
spatial_consistency: 0.88
manipulation_regions: []
texture_inconsistency: LOW
```

**해석**:
"공간적으로 일관성이 높으며(0.88) 조작 영역이 탐지되지 않았습니다.
이는 다음 중 하나를 의미합니다:
1. 자연 이미지 (조작 없음)
2. GAN/Diffusion으로 전체 생성된 이미지 (모든 영역이 동일 출처)

공간 분석만으로는 판단이 어렵습니다.
GAN 전체 생성 이미지는 '불일치'가 없으므로 공간 분석이 탐지하지 못합니다.
주파수 분석(격자 패턴), 노이즈 분석(PRNU), 워터마크 결과를 참고하세요."

---

## 🔍 특수 케이스

### 1. AI 기반 Seamless Blending
**도구**: Adobe Firefly, Generative Fill

**특징**:
- AI가 경계를 부드럽게 혼합
- 텍스처 불일치 최소화

**탐지**:
- ViT는 여전히 미세한 패턴 차이 탐지
- 다만 신뢰도가 낮을 수 있음 (0.6-0.7 수준)

### 2. 전체 이미지 리터칭
**예**: 피부 보정, 색상 조정

**공간 분석 결과**:
- spatial_consistency 여전히 높음
- 조작 영역 탐지 안 됨

**이유**:
- 전체에 균일하게 적용 → 불일치 없음
- 공간 분석은 "지역적 차이"를 찾음

### 3. 압축 아티팩트
**JPEG 압축**:
- 블록 경계(8x8)에서 불일치 발생 가능
- False Positive 위험

**대응**:
- ViT는 JPEG 블록 패턴 학습
- 실제 조작과 구별 가능

---

## 📖 참고문헌

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. Wu et al., "ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries", CVPR 2019
3. Guillaro et al., "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization", CVPR 2023

---

**핵심 원리**:
"자연 이미지는 하나의 이야기. 조작 이미지는 여러 이야기의 패치워크."
