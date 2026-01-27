# Tool Specialization 전략 보고서

**작성일**: 2026-01-26
**목적**: 각 툴의 전문성 강화 및 상호보완적 시스템 구축

---

## Executive Summary

### 현재 문제
- **기대**: 각 툴이 자신의 전문 영역에서 90%+ 정답률, 약점은 다른 툴이 보완
- **현실**: Frequency 47%, Noise 50% 정답률로 목표에 크게 미달
- **Gap**: 18%의 케이스는 두 툴 모두 오답

### 긍정적 발견
- **상호보완성 우수**: 45%의 complementary benefit
- **Union 커버리지**: 67% (둘 중 하나라도 맞춤)
- **툴 간 차별화**: Freq만/Noise만 맞춘 케이스가 각각 22-23%

### 전략적 방향
1. **Generalist → Specialist 전환**: 모든 이미지를 판정하려 하지 말고, 확실한 것만 판정
2. **Conservative Thresholding**: 불확실하면 UNCERTAIN으로 남김
3. **Domain-Specific Optimization**: 각 툴이 잘하는 이미지 유형 특화
4. **Confidence-Weighted Consensus**: 확신도 기반 가중 투표

---

## 1. 현재 상태 분석

### 1.1 개별 툴 성능

```
Frequency Tool:
  Correct:   47/100 (47.0%)
  Wrong:     41/100 (41.0%)  ← 문제!
  Uncertain: 12/100 (12.0%)

Noise Tool:
  Correct:   50/100 (50.0%)
  Wrong:     47/100 (47.0%)  ← 문제!
  Uncertain:  3/100 ( 3.0%)
```

**문제점**:
- 두 툴 모두 **오답이 정답보다 많거나 비슷**
- 특히 Noise tool은 UNCERTAIN을 거의 사용 안 함 (3%) → 과신

### 1.2 상호보완성 분석

```
Both Correct:    22/100 (22.0%) ← 두 툴 합의, 신뢰도 높음
Both Wrong:      18/100 (18.0%) ← 심각한 gap
Freq Only:       22/100 (22.0%) ← Frequency의 전문 영역
Noise Only:      23/100 (23.0%) ← Noise의 전문 영역
Both Uncertain:   0/100 ( 0.0%) ← 거의 없음

Union Coverage: 67/100 (67.0%)
  = Both Correct (22) + Freq Only (22) + Noise Only (23)

Complementary Benefit: 45% ← 매우 우수!
```

**해석**:
1. ✅ **상호보완성 검증됨**: 두 툴이 서로 다른 케이스를 잘 잡음
2. ⚠️ **합의 정확도 낮음**: Both Correct가 22%만
3. 🚨 **Gap 존재**: 18%는 둘 다 못 잡음 → 다른 툴(Spatial, EXIF) 필요

### 1.3 최악 케이스 분석 (Both Tools Wrong)

**AI 이미지를 자연으로 오판 (18개 중 대부분):**
```
005_biggan_00074.png: Freq 0.35, Noise 0.00
001_biggan_00035.png: Freq 0.35, Noise 0.00
006_biggan_00074.png: Freq 0.38, Noise 0.00
000_biggan_00074.png: Freq 0.38, Noise 0.00
001_biggan_00074.png: Freq 0.20, Noise 0.00
```

**특징**:
- 모두 **매우 낮은 점수** (0.0-0.4 범위)
- BigGAN이 생성한 **고품질 이미지**
- 주파수/노이즈 특성이 자연 이미지와 거의 동일
- **이런 케이스는 Spatial tool이나 EXIF metadata로 탐지해야 함**

---

## 2. 근본 원인 분석

### 2.1 왜 각 툴의 성능이 낮은가?

#### Frequency Tool (47%)

**성공하는 케이스** (47개):
- 명확한 GAN artifacts (grid, checkerboard)
- Power spectrum slope가 확실히 낮음 (< 1.2)
- High-frequency abnormality 높음

**실패하는 케이스** (41개):
- BigGAN 고품질 이미지 (자연스러운 주파수 분포)
- JPEG 압축된 자연 이미지 (slope 낮음, HF 높음)
- 중간 영역 (threshold 근처)

**문제점**:
- **Threshold가 aggressive**: ai_threshold=0.48은 너무 낮음
- **HF feature 과민**: 자연 이미지에 0.989 평균 반응
- **BigGAN 특성**: Checkerboard artifacts 약함

#### Noise Tool (50%)

**성공하는 케이스** (50개):
- MVSS-Net이 조작 패턴 탐지
- 또는 PRNU/SRM에서 AI 노이즈 패턴 발견

**실패하는 케이스** (47개):
- BigGAN 고품질 이미지 (자연스러운 노이즈)
- JPEG 압축으로 노이즈 정보 손실
- MVSS 점수가 0.0인 케이스들

**문제점**:
- **MVSS threshold 과신**: mvss_threshold=0.84는 매우 높음
- **UNCERTAIN 부족**: 3%만 불확실 판정 → 과신
- **JPEG 취약**: 압축으로 PRNU 패턴 손실

### 2.2 목표 vs 현실의 차이

**목표 (이상적 Multi-Agent 시스템)**:
```
각 툴의 역할:
- Frequency: GAN artifacts 명확한 케이스에서 95%+ 정답
- Noise:     조작/PRNU 패턴 명확한 케이스에서 95%+ 정답
- Spatial:   공간적 불일치 명확한 케이스에서 95%+ 정답
- EXIF:      메타데이터 이상 명확한 케이스에서 95%+ 정답

불확실한 케이스 → UNCERTAIN → 다른 툴에 의존
```

**현실**:
```
- Frequency: 모든 이미지 판정 시도 → 47% 정답
- Noise:     모든 이미지 판정 시도 → 50% 정답
- UNCERTAIN을 충분히 사용하지 않음
- "확실하지 않으면 UNCERTAIN" 원칙 미적용
```

---

## 3. 해결 전략

### 전략 1: Conservative Thresholding

**현재 문제**: Threshold가 aggressive → 불확실해도 판정

**해결**:
```python
# Before (Aggressive)
ai_threshold = 0.48
auth_threshold = 0.40
uncertain_margin = 0.0  # 불확실 구간 없음!

# After (Conservative)
ai_threshold = 0.60      # 0.6 이상만 AI로 판정
auth_threshold = 0.35    # 0.35 이하만 자연으로 판정
uncertain_margin = 0.25  # 0.35-0.60 구간은 UNCERTAIN
```

**기대 효과**:
- 확실한 케이스만 판정 → Precision 상승
- 불확실한 케이스는 UNCERTAIN → 다른 툴에 판단 위임
- **목표**: Correct 70%+, Wrong 10% 이하, Uncertain 20%

### 전략 2: Confidence-Based Filtering

**원칙**: "Low confidence는 UNCERTAIN으로 강제 변환"

```python
def analyze(self, image):
    # 기존 로직
    verdict = self._calculate_verdict(score)
    confidence = self._calculate_confidence(score, evidence)

    # 신규: Confidence 기반 필터링
    if confidence < 0.7:  # 70% 미만 확신은 불확실 처리
        verdict = Verdict.UNCERTAIN
        explanation = f"Low confidence ({confidence:.0%}). 다른 도구로 검증 필요."

    return ToolResult(verdict=verdict, confidence=confidence, ...)
```

**적용 대상**:
- Frequency Tool: confidence < 0.7 → UNCERTAIN
- Noise Tool: confidence < 0.8 → UNCERTAIN (현재 과신 경향)

### 전략 3: Domain-Specific Thresholds

**발견**: 이미지 유형마다 최적 threshold가 다름

**구현**:
```python
class FrequencyAnalysisTool:
    def _get_adaptive_threshold(self, evidence):
        """이미지 특성에 따라 동적 threshold"""

        # JPEG 이미지는 더 보수적 판정
        if evidence["is_likely_jpeg"]:
            return {
                "ai_threshold": 0.65,     # 더 높은 기준
                "auth_threshold": 0.30,
                "uncertain_margin": 0.35  # 더 넓은 불확실 구간
            }

        # PNG 이미지는 현재 threshold
        else:
            return {
                "ai_threshold": 0.60,
                "auth_threshold": 0.35,
                "uncertain_margin": 0.25
            }
```

### 전략 4: Evidence-Based Confidence

**현재 문제**: Confidence 계산이 단순함

**개선**:
```python
def _calculate_confidence(self, score, evidence):
    """증거 기반 신뢰도 계산"""

    base_confidence = abs(score - 0.5) * 2  # 0.5에서 멀수록 높음

    # 감점 요인
    penalties = 0.0

    # 1. 특징 간 불일치
    features = [
        evidence["grid_score"],
        evidence["checkerboard_score"],
        evidence["slope_score"],
        evidence["hf_abnormality"]
    ]
    feature_std = np.std(features)
    if feature_std > 0.4:  # 특징 간 차이 크면 불확실
        penalties += 0.2

    # 2. JPEG 압축 (신뢰도 낮아짐)
    if evidence["is_likely_jpeg"]:
        penalties += 0.15

    # 3. Threshold 근처
    if 0.45 < score < 0.55:
        penalties += 0.3

    confidence = max(0.0, base_confidence - penalties)
    return confidence
```

### 전략 5: Two-Stage Filtering

**원칙**: "약한 특징은 강한 특징이 뒷받침할 때만 사용"

```python
def _two_stage_verdict(self, evidence):
    """2단계 필터링: 강한 증거 먼저, 약한 증거는 보조"""

    # Stage 1: 강한 특징만으로 판정
    strong_features = {
        "slope": evidence["slope_score"],      # 가장 강력
        "checkerboard": evidence["checkerboard_score"]
    }

    strong_score = (
        0.6 * strong_features["slope"] +
        0.4 * strong_features["checkerboard"]
    )

    # 강한 특징만으로 확실하면 즉시 판정
    if strong_score > 0.75:
        return Verdict.AI_GENERATED, 0.9
    elif strong_score < 0.25:
        return Verdict.AUTHENTIC, 0.9

    # Stage 2: 약한 특징 추가 (보조만)
    weak_features = {
        "grid": evidence["grid_score"],
        "hf": evidence["hf_abnormality"]
    }

    # 강한 특징 기반 + 약한 특징 보조
    final_score = (
        0.7 * strong_score +
        0.3 * np.mean(list(weak_features.values()))
    )

    # Conservative threshold
    if final_score > 0.65:
        return Verdict.AI_GENERATED, 0.7
    elif final_score < 0.35:
        return Verdict.AUTHENTIC, 0.7
    else:
        return Verdict.UNCERTAIN, 0.5
```

---

## 4. 구체적 실행 계획

### Phase 1: Conservative Thresholding (즉시 적용)

**Frequency Tool 수정**:
```python
# configs/tool_thresholds.json
{
  "frequency": {
    "ai_threshold": 0.60,        # 0.48 → 0.60
    "auth_threshold": 0.35,      # 0.40 → 0.35
    "uncertain_margin": 0.25,    # 0.0 → 0.25
    "min_confidence": 0.70       # 신규
  }
}
```

**Noise Tool 수정**:
```python
{
  "noise": {
    "mvss_threshold": 0.75,           # 0.84 → 0.75
    "mvss_auth_threshold": 0.50,      # 0.84 → 0.50
    "mvss_uncertain_margin": 0.25,    # 0.0 → 0.25
    "min_confidence": 0.75            # 신규
  }
}
```

**예상 결과**:
- Correct: 47% → 50-55%
- Wrong: 41% → 15-20%
- Uncertain: 12% → 30-35%
- **목표**: "틀리는 것보다 불확실이 낫다"

### Phase 2: Confidence-Based Filtering (1주)

**구현**:
1. `_calculate_confidence()` 함수 개선
2. Low confidence → UNCERTAIN 강제 변환
3. Evidence-based penalty 시스템

**검증**:
- 100개 샘플에서 confidence distribution 분석
- Wrong 케이스의 평균 confidence 측정
- Threshold tuning

### Phase 3: Domain-Specific Optimization (2주)

**분석**:
1. JPEG vs PNG 성능 차이 측정
2. 이미지 해상도별 성능 측정
3. 장면 유형별 (얼굴/풍경/사물) 성능 측정

**구현**:
- Adaptive threshold 시스템
- 이미지 특성 자동 감지
- 특성별 최적 threshold 적용

### Phase 4: Two-Stage Filtering (2주)

**구현**:
- Strong features 우선 판정
- Weak features 보조 활용
- Feature conflict detection

**검증**:
- Feature agreement rate 측정
- Conflict 케이스 분석

---

## 5. 성공 지표

### 개별 툴 목표 (Conservative 모드)

**Frequency Tool**:
```
Correct:   60%+ (현재 47%)
Wrong:     15% 이하 (현재 41%) ← 핵심 목표
Uncertain: 25-30% (현재 12%)

Precision: 80%+ (확실할 때만 판정)
Recall:    60%+
```

**Noise Tool**:
```
Correct:   65%+ (현재 50%)
Wrong:     15% 이하 (현재 47%) ← 핵심 목표
Uncertain: 20-25% (현재 3%)

Precision: 85%+
Recall:    65%+
```

### 시스템 목표 (Multi-Tool Consensus)

```
Union Coverage: 80%+ (현재 67%)
  - Both Correct: 40%+
  - Complementary: 40%+

Both Wrong: 10% 이하 (현재 18%)

High Confidence Agreement: 90%+ precision
  (두 툴 모두 확신할 때 정답률)
```

---

## 6. 장기 전략: Specialization by Image Type

### 각 툴의 전문 영역 재정의

#### Frequency Tool 전문 영역
**Best Cases** (Precision 90%+):
- GAN artifacts 명확한 이미지
  - StyleGAN, ProGAN (체커보드 강함)
  - Low-quality GAN outputs
- PNG 형식 (압축 없음)
- 고해상도 이미지

**Weak Cases** (UNCERTAIN 처리):
- JPEG 고압축 이미지
- BigGAN 고품질 출력
- 저해상도 이미지 (< 256px)

#### Noise Tool 전문 영역
**Best Cases** (Precision 90%+):
- 조작된 이미지 (MVSS-Net)
- 카메라 센서 패턴 명확한 이미지
- 미압축 또는 저압축 이미지

**Weak Cases** (UNCERTAIN 처리):
- JPEG 고압축 이미지
- 스마트폰 후처리 이미지
- AI 생성 + JPEG 재압축

#### Spatial Tool 전문 영역
**Best Cases** (Precision 90%+):
- 조작 영역 명확한 이미지
- Inpainting, splicing
- 경계선 불일치

**Weak Cases** (UNCERTAIN 처리):
- 자연스러운 이미지
- 전체가 AI 생성된 이미지 (조작 영역 없음)

#### EXIF Tool 전문 영역
**Best Cases** (Precision 95%+):
- 메타데이터 부재
- Software tag에 "AI" 포함
- 날짜/위치 불일치

**Weak Cases** (UNCERTAIN 처리):
- EXIF가 정상적으로 보이는 이미지
- 메타데이터 재작성된 이미지

---

## 7. COBRA Consensus 전략

### 현재 상태
- COBRA 알고리즘 구현되어 있으나 미사용
- 단순 majority voting 사용

### COBRA 활성화 전략

**Confidence-Weighted Consensus**:
```python
def cobra_consensus(tool_results):
    """신뢰도 기반 COBRA 합의"""

    # 1. High confidence 툴만 투표권
    high_conf_results = [
        r for r in tool_results
        if r.confidence >= 0.8 and r.verdict != Verdict.UNCERTAIN
    ]

    if not high_conf_results:
        # 모두 낮은 확신 → UNCERTAIN
        return Verdict.UNCERTAIN, 0.5, "모든 도구가 불확실"

    # 2. Confidence-weighted voting
    ai_weight = sum(
        r.confidence for r in high_conf_results
        if r.verdict == Verdict.AI_GENERATED
    )
    auth_weight = sum(
        r.confidence for r in high_conf_results
        if r.verdict == Verdict.AUTHENTIC
    )

    # 3. 판정
    if ai_weight > auth_weight * 1.5:  # AI 쪽이 1.5배 이상 강해야
        return Verdict.AI_GENERATED, ai_weight / len(high_conf_results), ...
    elif auth_weight > ai_weight * 1.5:
        return Verdict.AUTHENTIC, auth_weight / len(high_conf_results), ...
    else:
        return Verdict.UNCERTAIN, 0.5, "도구 간 의견 불일치"
```

**Domain Router**:
```python
def route_to_specialist(image_metadata):
    """이미지 특성에 따라 전문 툴에 가중치"""

    weights = {
        "frequency": 1.0,
        "noise": 1.0,
        "spatial": 1.0,
        "exif": 1.0
    }

    # PNG 이미지 → Frequency/Noise 강화
    if image_metadata["format"] == "PNG":
        weights["frequency"] *= 1.5
        weights["noise"] *= 1.5

    # JPEG 이미지 → EXIF/Spatial 강화
    elif image_metadata["format"] == "JPEG":
        weights["exif"] *= 1.5
        weights["spatial"] *= 1.2
        weights["frequency"] *= 0.7  # 약화

    # 메타데이터 풍부 → EXIF 강화
    if image_metadata["has_rich_exif"]:
        weights["exif"] *= 2.0

    return weights
```

---

## 8. 결론

### 핵심 인사이트

1. **현재 문제의 본질**:
   - 각 툴이 "모든 이미지 판정" 시도 → 50% 정답률
   - "확실한 영역에서만 판정" 원칙 부재

2. **해결의 핵심**:
   - **Conservative Thresholding**: 확실할 때만 판정
   - **UNCERTAIN 적극 사용**: 불확실하면 다른 툴에 위임
   - **Specialization**: 각 툴의 전문 영역 명확화

3. **상호보완성은 검증됨**:
   - 45%의 complementary benefit
   - 툴 간 차별화 우수
   - Multi-agent 구조의 가치 입증

### 즉시 실행 항목

**High Priority**:
1. ✅ Conservative threshold 적용 (오늘)
2. ✅ Confidence-based filtering (1주)
3. ✅ UNCERTAIN 확대 전략 (1주)

**Medium Priority**:
4. Domain-specific optimization (2주)
5. COBRA consensus 활성화 (2주)
6. Two-stage filtering (2주)

**Long-term**:
7. 다양한 GAN 모델 검증 (1개월)
8. Real-world dataset 테스트 (1개월)
9. 딥러닝 기반 특징 추가 (2-3개월)

### 기대 효과

**단기** (Conservative threshold 적용 후):
- Wrong: 41% → 15% (✓ 핵심 목표)
- Correct: 47% → 55-60%
- Uncertain: 12% → 25-30%

**중기** (모든 전략 적용 후):
- Union Coverage: 67% → 80%+
- Both Wrong: 18% → < 10%
- High Confidence Precision: 90%+

**장기** (System-wide optimization):
- 각 툴의 전문 영역에서 90%+ precision
- Multi-agent consensus로 95%+ system precision
- Real-world 적용 가능 수준

---

**Report ID**: TOOL-SPEC-20260126
**Next Action**: Conservative thresholding 즉시 적용
**Review Date**: 1주 후 성능 재측정
