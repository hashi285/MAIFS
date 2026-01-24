# 노이즈 분석 도메인 지식
**PRNU (Photo Response Non-Uniformity) Noise Analysis**

---

## 📚 과학적 근거

### 핵심 논문
1. **"Digital camera identification from sensor pattern noise" (Lukas et al., 2006)**
   - PRNU는 카메라 센서의 고유한 "지문"
   - 각 픽셀의 감광 민감도 차이로 인한 노이즈

2. **"Noiseprint: a CNN-based camera model fingerprint" (Cozzolino et al., 2019)**
   - 딥러닝 기반 노이즈 패턴 추출
   - 카메라 모델별 고유 노이즈 학습

3. **"Camera model identification using sensor noise" (Chen et al., 2008)**
   - PRNU의 주파수 특성
   - AI 생성 이미지는 PRNU 없음

---

## 🔬 분석 원리

### 1. PRNU란?
```
카메라 센서의 제조 과정:
  실리콘 웨이퍼 → 포토다이오드 어레이 → 미세한 불균일성
```

**물리적 원인**:
- 각 픽셀의 감광 민감도가 미세하게 다름
- 제조 과정의 불완전성 (균일하게 만들 수 없음)
- 이 패턴은 **카메라마다 고유함** (지문과 같음)

**수학적 모델**:
```
I(x, y) = I₀(x, y) × [1 + K(x, y)] + Θ(x, y)

I: 관측 이미지
I₀: 실제 장면
K: PRNU 패턴 (카메라 고유)
Θ: 기타 노이즈 (shot noise, read noise)
```

### 2. PRNU 추출 방법
```
이미지 → Denoising Filter → 노이즈 잔여물 → PRNU 패턴
```

**사용 필터**:
- Wiener Filter (전통적 방법)
- BM3D (고급 denoising)
- DnCNN (딥러닝 기반)

**추출된 PRNU의 특징**:
- 고주파 성분 (카메라 센서의 미세 패턴)
- 이미지 내용과 독립적
- 동일 카메라로 찍은 모든 사진에 공통

---

## 📊 메트릭 해석 가이드

### 1. prnu_consistency (PRNU 일관성)
**측정 방법**: 이미지 전체에서 PRNU 패턴의 일관성 측정

| 범위 | 의미 | 과학적 근거 |
|------|------|------------|
| 0.8 - 1.0 | 매우 일관적 | 실제 카메라 촬영. PRNU 패턴 명확 |
| 0.5 - 0.8 | 중간 일관성 | 압축/편집된 실제 사진 또는 혼합 이미지 |
| 0.0 - 0.5 | 일관성 없음 | AI 생성 또는 심하게 조작된 이미지 |

**AI 생성 이미지의 특징**:
- GAN/Diffusion은 PRNU를 "학습"하지 않음
- 훈련 데이터의 PRNU는 모두 다르므로 학습 불가
- 결과: PRNU 패턴이 없거나 무작위

### 2. noise_pattern_presence (노이즈 패턴 존재 여부)
**측정 방법**: 카메라 센서 노이즈 패턴 탐지

**의미**:
- `True`: 카메라 고유 노이즈 검출
- `False`: 노이즈 없음 또는 인위적 노이즈

**판별 기준**:
```python
if correlation(extracted_noise, natural_prnu_template) > threshold:
    → AUTHENTIC (실제 카메라)
else:
    → AI_GENERATED or MANIPULATED
```

### 3. sensor_fingerprint_match (센서 지문 매칭)
**측정 방법**: 알려진 카메라 모델의 PRNU와 비교

| 값 | 의미 | 설명 |
|-----|------|------|
| 0.9 - 1.0 | 특정 카메라 일치 | 카메라 모델 식별 가능 |
| 0.6 - 0.9 | 유사 카메라 | 같은 제조사/모델 계열 |
| < 0.6 | 매칭 실패 | AI 생성 또는 알려지지 않은 카메라 |

**주의사항**:
- 새로운 카메라 모델은 데이터베이스에 없을 수 있음
- "매칭 실패 ≠ 무조건 AI 생성"

---

## ⚖️ 분석의 강점과 한계

### 강점
✅ **물리적 근거**
   - PRNU는 카메라 하드웨어의 물리적 특성
   - 소프트웨어로 위조 매우 어려움

✅ **AI 생성 이미지 확실히 구별**
   - GAN/Diffusion은 PRNU 생성 불가
   - PRNU 있음 → 실제 카메라 촬영 확실

✅ **조작 탐지**
   - 복사-붙여넣기 → PRNU 불일치
   - 지역별 PRNU 분석으로 조작 영역 탐지

### 한계
❌ **압축/편집에 취약**
   - JPEG 압축 → PRNU 약화
   - 강한 필터링 → PRNU 손상

❌ **AI+실제 혼합 이미지**
   - 실제 사진에 GAN으로 객체 추가
   - 배경: PRNU 있음, 추가된 객체: PRNU 없음
   - → 전체 평균 시 혼란

❌ **계산 비용**
   - PRNU 추출은 계산 집약적
   - 고해상도 이미지는 느림

---

## 🤝 다른 분석과의 관계

### vs Frequency Analysis
**보완 관계**:
- Frequency: 생성 방법의 흔적 (격자 패턴)
- Noise: 촬영 기기의 흔적 (센서 지문)

**상충 시 해석**:
```
Frequency: AI_GENERATED (격자 패턴 있음)
Noise: AUTHENTIC (PRNU 있음)

→ 해석: 실제 사진에 GAN으로 객체/배경을 합성한 혼합 이미지
→ 추천: 공간 분석으로 조작 영역 특정
```

### vs Watermark Detection
**독립적 관계**:
- PRNU: 카메라 출처
- Watermark: 생성 모델 출처

**함께 사용**:
```
PRNU 없음 + Stable Diffusion 워터마크
→ Stable Diffusion 생성 확실
```

### vs Spatial Analysis
**보완 관계**:
- Noise: 전역 일관성 (global consistency)
- Spatial: 지역 불일치 (local inconsistency)

**조작 탐지 시**:
```
1. PRNU로 조작 여부 판단
2. Spatial로 조작 위치 특정
```

---

## 💡 해석 예시

### Case 1: 실제 카메라 촬영
```
prnu_consistency: 0.87
noise_pattern_presence: True
sensor_fingerprint_match: 0.92 (Canon EOS 5D)
```

**해석**:
"명확한 PRNU 패턴이 검출되었습니다(일관성 0.87).
카메라 센서의 고유한 노이즈 지문이 이미지 전체에서 일관되게 나타나며,
Canon EOS 5D Mark IV의 알려진 PRNU 패턴과 92% 일치합니다.
이는 실제 카메라로 촬영된 원본 이미지임을 강력히 시사합니다."

### Case 2: AI 생성 이미지
```
prnu_consistency: 0.12
noise_pattern_presence: False
sensor_fingerprint_match: 0.05
```

**해석**:
"PRNU 패턴이 거의 검출되지 않았습니다(일관성 0.12).
카메라 센서의 고유 노이즈가 존재하지 않으며,
어떤 알려진 카메라 모델과도 매칭되지 않습니다.
이는 GAN 또는 Diffusion 모델로 생성된 이미지의 전형적인 특징입니다.
AI 생성 모델은 물리적 카메라 센서가 없으므로 PRNU를 생성할 수 없습니다."

### Case 3: 혼합 이미지 (실제 + AI)
```
prnu_consistency: 0.54
noise_pattern_presence: True (부분적)
spatial_variance: HIGH (지역별 차이 큼)
```

**해석**:
"PRNU 일관성이 중간 수준(0.54)이며, 이미지의 일부 영역에서만
노이즈 패턴이 검출됩니다. 이는 실제 사진에 AI로 생성된 객체를
합성한 혼합 이미지일 가능성을 시사합니다.
배경: PRNU 있음 (실제 촬영)
전경 객체: PRNU 없음 (AI 생성)
공간 분석으로 정확한 조작 영역 특정이 필요합니다."

---

## 🔍 특수 케이스

### 1. 스마트폰 카메라
**특징**:
- 강한 후처리 (HDR, AI 필터)
- PRNU가 약화될 수 있음
- 하지만 완전히 사라지지는 않음

**판단**:
- prnu_consistency < 0.7이어도 AUTHENTIC 가능
- noise_pattern_presence가 더 중요

### 2. 스크린샷
**특징**:
- PRNU 없음 (카메라로 찍지 않음)
- 하지만 AI 생성도 아님

**판단**:
- PRNU만으로는 판단 불가
- 다른 증거 필요 (워터마크, 주파수 등)

### 3. RAW vs JPEG
**RAW**:
- PRNU 최대한 보존
- 가장 정확한 분석 가능

**JPEG**:
- 압축으로 PRNU 약화
- 하지만 여전히 탐지 가능 (완전 소멸은 아님)

---

## 📖 참고문헌

1. Lukas et al., "Digital Camera Identification from Sensor Pattern Noise", IEEE TIFS 2006
2. Cozzolino et al., "Noiseprint: A CNN-Based Camera Model Fingerprint", IEEE TIFS 2019
3. Chen et al., "Determining Image Origin and Integrity Using Sensor Noise", IEEE TIFS 2008
4. Goljan et al., "Digital Camera Identification from Images - Estimating False Acceptance Probability", IWDW 2008

---

**핵심 원리**:
"자연의 불완전성(센서의 미세한 차이)이 AI가 모방할 수 없는 진정성의 증거"
