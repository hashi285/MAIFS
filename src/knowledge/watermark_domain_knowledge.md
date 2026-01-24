# 워터마크 탐지 도메인 지식
**AI Watermark Detection (OmniGuard Framework)**

---

## 📚 과학적 근거

### 핵심 논문
1. **"OmniGuard: Universal Watermark Detection for Generative AI" (2024)**
   - 다양한 AI 모델의 워터마크 통합 탐지
   - Stable Diffusion, DALL-E, Midjourney 등

2. **"The Stable Signature: Rooting Watermarks in Latent Diffusion Models" (2023)**
   - Diffusion 모델에 내장된 워터마크
   - Decoder 레벨 워터마크 삽입

3. **"Tree-Ring Watermarks" (2024)**
   - Diffusion 과정에 워터마크 내장
   - 초기 노이즈 패턴 기반

---

## 🔬 분석 원리

### 1. 워터마크란?
```
생성 AI → 이미지 생성 시 고유 패턴 삽입 → 워터마크
```

**목적**:
- AI 생성 이미지 출처 추적
- 저작권 보호
- 딥페이크 방지

**유형**:
1. **Visible Watermark**: 육안으로 보이는 로고/텍스트
2. **Invisible Watermark**: 사람은 못 보지만 알고리즘은 탐지 가능
3. **Latent Watermark**: 생성 과정에 내장

### 2. OmniGuard 탐지 원리
```
이미지 → Feature Extraction (CNN) → Watermark Decoder → 확률
```

**훈련 데이터**:
- Stable Diffusion v1.4, v1.5, v2.0, v2.1
- DALL-E 2, DALL-E 3
- Midjourney v4, v5
- Adobe Firefly

**탐지 메커니즘**:
- 각 모델의 워터마크 패턴 학습
- 다중 디코더 앙상블
- 모델별 신뢰도 출력

---

## 📊 메트릭 해석 가이드

### 1. watermark_detected (워터마크 탐지 여부)
**측정 방법**: OmniGuard 디코더 출력

**의미**:
- `True`: AI 모델의 워터마크 검출
- `False`: 워터마크 없음

**주의**:
- `False` ≠ 자연 이미지
- 워터마크 없는 AI 모델도 많음 (오픈소스 모델 대부분)

### 2. watermark_confidence (워터마크 신뢰도)
| 범위 | 의미 | 행동 |
|------|------|------|
| 0.9 - 1.0 | 확실한 워터마크 | AI 생성 확정 |
| 0.7 - 0.9 | 높은 확률 | AI 생성 가능성 높음 |
| 0.5 - 0.7 | 애매함 | 추가 분석 필요 |
| < 0.5 | 워터마크 없음 | 판단 불가 |

### 3. detected_model (탐지된 모델)
**가능한 값**:
- `stable_diffusion_v1.5`
- `stable_diffusion_v2.1`
- `dalle2`
- `dalle3`
- `midjourney_v5`
- `unknown`

**의미**:
- 구체적 생성 모델 식별
- 출처 추적 가능

---

## ⚖️ 분석의 강점과 한계

### 강점
✅ **구체적 출처 정보**
   - 어떤 AI 모델로 생성했는지 식별
   - 주파수/노이즈 분석은 "AI 생성 여부"만 판단

✅ **워터마크 있으면 100% 확실**
   - False Positive 매우 낮음
   - 워터마크는 의도적으로 삽입된 것

✅ **압축에 강건**
   - 워터마크는 강건하게 설계됨
   - JPEG 압축 후에도 탐지 가능

### 한계
❌ **워터마크 없는 모델 탐지 불가**
   - 오픈소스 Stable Diffusion (워터마크 제거 가능)
   - ComfyUI, Automatic1111 등

❌ **워터마크 제거 공격**
   - Adversarial perturbation
   - 재훈련으로 워터마크 제거 가능

❌ **커버리지 제한**
   - 훈련된 모델만 탐지 가능
   - 새로운 AI 모델은 업데이트 필요

---

## 🤝 다른 분석과의 관계

### vs Frequency Analysis
**보완 관계**:
```
Frequency: AI_GENERATED (격자 패턴) → "GAN으로 생성됨"
Watermark: stable_diffusion_v2.1 → "Stable Diffusion v2.1로 생성됨"
```

**함께 사용**:
- Frequency가 "AI 생성 여부" 판단
- Watermark가 "구체적 모델" 식별

### vs Noise Analysis (PRNU)
**독립적**:
- PRNU: 카메라 흔적
- Watermark: AI 모델 흔적

**상충 불가**:
- PRNU 없음 + Watermark 있음 → AI 생성 확실
- PRNU 있음 + Watermark 있음 → 불가능 (물리적 모순)

### vs Spatial Analysis
**보완 관계**:
- Watermark: 전역 정보 (이미지 전체가 AI 생성)
- Spatial: 지역 정보 (특정 부분만 AI 생성)

---

## 💡 해석 예시

### Case 1: Stable Diffusion 확실
```
watermark_detected: True
watermark_confidence: 0.94
detected_model: stable_diffusion_v2.1
```

**해석**:
"Stable Diffusion v2.1의 워터마크가 94% 신뢰도로 탐지되었습니다.
이 이미지는 Stable Diffusion v2.1 모델로 생성된 것이 확실합니다.
워터마크는 생성 과정에서 의도적으로 삽입되며,
자연 이미지에서는 절대 나타나지 않는 패턴입니다."

### Case 2: 워터마크 없음 (판단 불가)
```
watermark_detected: False
watermark_confidence: 0.15
detected_model: None
```

**해석**:
"알려진 AI 모델의 워터마크가 탐지되지 않았습니다.
이는 다음 중 하나를 의미할 수 있습니다:
1. 자연 이미지 (실제 카메라 촬영)
2. 워터마크 없는 AI 모델 (오픈소스 Stable Diffusion 등)
3. 워터마크가 제거된 AI 생성 이미지
워터마크만으로는 판단할 수 없으므로 주파수, 노이즈 분석 결과가 필요합니다."

### Case 3: 낮은 신뢰도 (경계 케이스)
```
watermark_detected: True
watermark_confidence: 0.68
detected_model: midjourney_v5
```

**해석**:
"Midjourney v5 워터마크가 탐지되었으나 신뢰도가 중간 수준(68%)입니다.
가능한 원인:
- 강한 JPEG 압축으로 워터마크 약화
- 후처리(필터, 리사이징)로 워터마크 손상
- False Positive 가능성
다른 분석 결과(주파수, 공간)와 교차 검증이 필요합니다."

---

## 🔍 특수 케이스

### 1. 오픈소스 모델
**문제**:
- Stable Diffusion 오픈소스 버전은 워터마크 제거 가능
- ComfyUI, Automatic1111 등

**대응**:
- 워터마크 탐지 실패 ≠ 자연 이미지
- 다른 분석 필수

### 2. 워터마크 제거 공격
**방법**:
- Adversarial perturbation
- 재훈련 (fine-tuning)

**탐지**:
- 제거 흔적을 주파수/공간 분석으로 탐지 가능

### 3. 혼합 이미지
**시나리오**:
- 실제 사진 + AI 생성 객체

**워터마크 동작**:
- 전체 이미지에서 워터마크 탐지 시도
- 부분적 워터마크는 신뢰도 낮게 나타남

---

## 📖 참고문헌

1. OmniGuard Team, "Universal Watermark Detection Framework", 2024
2. Fernandez et al., "The Stable Signature", arXiv 2023
3. Wen et al., "Tree-Ring Watermarks", ICML 2024

---

**핵심**:
"워터마크는 AI 모델의 '서명'. 발견되면 확실하지만, 없다고 자연 이미지는 아님."
