# SHIELD 연구 계획

> **Shapley-based Hardware-aware Interaction-preserving Ensemble Lightweighting for on-Device forensics**
>
> DAAC (KIPS 2026) 후속 연구. AGENTS.md §4에서 참조.

---

## 1. 연구 배경 및 목표

### 1.1 문제 정의

MAIFS의 DAAC는 서버 환경에서 Macro-F1 0.8613을 달성했으나, 4개 에이전트 합계 **~1.26GB / ~313ms**로 엣지 디바이스 배포가 불가능하다.

핵심 질문:
1. 4개 에이전트 중 **어떤 조합이 최적**인가? (단순 성능이 아닌 상호작용 기반)
2. 각 에이전트의 백본을 **얼마나 경량화**할 수 있는가? (포렌식 신호 보존 제약)
3. **Cascade 아키텍처**로 평균 추론 비용을 얼마나 줄일 수 있는가?

### 1.2 목표 스펙

| 지표 | 서버 (현재) | 엣지 (목표) |
|------|-----------|-----------|
| 모델 총 크기 | ~1.26GB | <500MB (이상적 ~180MB) |
| 추론 시간 (이미지 1장) | ~313ms | <1초 (RPi5 CPU) |
| Macro-F1 | 0.8613 | >0.80 (5%p 이내 드롭) |
| 메모리 사용 | ~4GB+ | <4GB (RPi5 8GB의 50%) |
| 타겟 디바이스 | GPU server | Raspberry Pi 5 (8GB) |

### 1.3 타겟 디바이스 선정 근거

**Raspberry Pi 5** 선택, Galaxy S26 배제:
- RPi5: 오픈 하드웨어, 재현성 보장, Hailo-8L NPU 확장 가능, 학술 연구 표준
- Galaxy S26: 폐쇄 플랫폼, NNAPI/SNPE 종속, 재현성 부족

RPi5 사양:
- CPU: Broadcom BCM2712, Cortex-A76 4코어 @ 2.4GHz
- RAM: 8GB LPDDR4X
- NPU (선택): Hailo-8L (13 TOPS INT8)
- OS: Raspberry Pi OS (64-bit, Debian-based)

---

## 2. 딥리서치 종합 결과

> 6개 PDF(Prompt 1~3, Gemini+GPT 각각) 분석 종합.
> 원본: `/data/jj812_files/DeepResearch_Prompt{1,2,3}_{Gemini,GPT}.pdf`

### 2.1 Prompt 1: On-Device 포렌식 SOTA

**경량 모델 대안**:
- **ForMa** (VMamba 기반): 37M params, 42G FLOPs — 중형 대안
- **MobileCLIP** (CVPR 2024): S0/S1/S2 variants, DataCompDR 학습 — FatFormer 백본 교체 최유력
- **RelayFormer**: progressive token compression — 포렌식 적용 시 주의 필요
- **SAPN/LaDeDa/Tiny-LaDeDa**: 경량 조작 탐지 모델군
- **BNN 기반 deepfake detection**: 극단적 경량화 사례

**Cascade 선행연구**:
- **CoE (Cascade of Experts)**: 7x 비용 절감 달성
- **NoScope**: 저비용 프록시 → 고비용 전문가 순차 호출
- **BranchyNet**: early-exit 네트워크
- **Chameleon**: adaptive configuration per input

**RPi5 실측 벤치마크** (GPT PDF):
- 경량 CNN (MobileNetV3 등): **~100 FPS** (CPU)
- 하이브리드 ViT: **~10 FPS** (CPU)
- MobileNetV3-Small: **~19ms/img** (CPU)

### 2.2 Prompt 2: Agent Selection Theory

**Model Shapley** (Data Shapley가 아님):
- N=4일 때 exact computation 가능 (2⁴=16 부분집합 전수 평가)
- 각 에이전트의 marginal contribution 정량화
- 참조: "Don't Always Pick the Highest-Performing Model" (arXiv 2602.08003)

**STII (Shapley-Taylor Interaction Index)**:
- k=2: pairwise interaction 측정
- Freq↔FatFormer 시너지를 수학적으로 정량화
- Faith-Shap과 대비하여 해석 용이

**PID (Partial Information Decomposition)**:
- 정보론적 관점에서 에이전트 출력 분해:
  - **Unique**: 해당 에이전트만 제공하는 정보
  - **Redundant**: 여러 에이전트가 중복 제공하는 정보
  - **Synergistic**: 조합에서만 나타나는 정보
- Freq↔FatFormer의 synergistic information이 높을 것으로 예상

**CKA (Centered Kernel Alignment)**:
- 사전학습된 에이전트 feature map 간 유사도
- 높은 CKA = 중복 → pruning 후보

**최적화 프레임워크**:
- **CoAI**: Shapley + knapsack 조합 최적화
- **CCIEP**: interaction-aware pruning
- **DREP**: diversity-regularized ensemble pruning
- Submodularity ratio 분석 → Freq↔FatFormer가 submodularity 위반 가능성

**핵심 인사이트**:
- `disagree_frequency_fatformer` = 56.5% feature importance는 **pairwise interaction이 individual contribution보다 중요**함을 의미
- 단순 greedy selection은 이 시너지를 파괴할 수 있음
- → **interaction-preserving constraint**가 필수

### 2.3 Prompt 3: 포렌식 모델 경량화

**양자화 민감도 (가장 중요한 발견)**:
- 포렌식 신호(PRNU, DCT 계수)는 **low-magnitude, high-frequency**
- INT8 양자화 노이즈가 신호 자체를 초과할 수 있음
- **PTQ(Post-Training Quantization)만으로는 불충분** → **QAT(Quantization-Aware Training) 필수**
- **Mixed-precision 전략**:
  - 입력단 노이즈 추출 레이어: **FP16 유지**
  - 깊은 시맨틱 레이어: **INT8 허용**
- 참조: FIMA-Q (ViT PTQ), block-wise pruning for steganalysis

**Pruning 전략**:
- L1-norm pruning: 포렌식에서 파괴적 → **사용 금지**
- **Taylor Expansion pruning**: gradient 기반 중요도 → 추천
- **Geometric Median pruning**: 필터 다양성 보존 → 추천
- **Token pruning**: 조작 영역 토큰 제거 위험 → **포렌식에서 사용 금지**
- Structured pruning: channel/filter 단위, 하드웨어 친화적

**FatFormer FAA 확인 (결정적 발견)**:
- FatFormer 논문 ablation: **ViT-B/16, Swin-B, Swin-L** 모두에서 FAA 어댑터 동작
- → FAA는 **backbone-agnostic** → MobileCLIP-S2로 교체 후 FAA만 재학습 가능
- 이것이 전체 경량화에서 **단일 최대 절감** (890MB → ~50MB)

**에이전트별 경량화 상세**:

| Agent | 전략 | 핵심 기법 | 절감 | 리스크 |
|-------|------|----------|------|--------|
| FatFormer | 백본 교체 | MobileCLIP-S2 + FAA retrain | 890→50MB | adapter transfer 실패 시 -2.5%p 이상 |
| CAT-Net | Structured pruning | Taylor/GeomMedian, DCT+RGB 양 stream 보존 | 150→55MB | DCT stream pruning 시 압축 탐지 무력화 |
| MVSS-Net | Teacher-student KD | MobileNetV3-Small + feature-level distillation | 120→25MB | edge supervision 소실 시 localization 성능 급락 |
| Mesorch | 백본 교체+pruning | Mesorch-P + Fast-SCNN | 100→50MB | SRM/DWT custom ops NPU 미지원 |

**배포 파이프라인**:
- CPU 경로: **ONNX Runtime + XNNPACK** 백엔드
- NPU 경로 (선택): **Hailo DFC → HEF 변환**
- 주의: DWT/SRM 같은 custom ops는 NPU 미지원 → CPU fallback
- Re-MTKD (Multi-Teacher KD): 4 teacher → 1 student 가능하지만 complexity 높음

---

## 3. 실험 설계

### Phase 1: Agent Valuation

**목적**: 4개 에이전트의 가치·상호작용을 이론적으로 정량화하여 최적 부분집합 결정

**실험 1.1: Model Shapley (Exact)**
```
입력: 기존 MAIFS 실데이터 (experiments/results/paper_final/ 기준)
방법: 2⁴=16개 부분집합 S ⊆ {Freq, Noise, FatFormer, Spatial}
      각 S에 대해 DAAC 메타 분류기 재학습 → Macro-F1 측정
      Shapley value φᵢ = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [v(S∪{i}) - v(S)]
출력: φ_freq, φ_noise, φ_fatformer, φ_spatial
```

**실험 1.2: STII (k=2 Pairwise Interaction)**
```
방법: 동일 16개 부분집합 결과 활용
      STII_{ij} = Σ_{S⊆N\{i,j}} [...] × [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]
출력: 6개 pairwise interaction 값 (특히 STII_freq_fatformer)
```

**실험 1.3: CKA Analysis**
```
방법: 4개 에이전트의 penultimate layer feature 추출 (동일 이미지셋)
      CKA(X_i, X_j) = HSIC(X_i, X_j) / √(HSIC(X_i, X_i) × HSIC(X_j, X_j))
출력: 4×4 CKA similarity matrix
```

**실험 1.4: PID (Partial Information Decomposition)**
```
방법: 각 에이전트 출력의 mutual information 분해
      I(Y; X_i, X_j) = Unique(X_i) + Unique(X_j) + Redundancy + Synergy
출력: 에이전트 쌍별 unique/redundant/synergistic 비율
```

### Phase 2: Model Compression

**실험 2.1: FatFormer → MobileCLIP-S2**
```
단계:
  1. MobileCLIP-S2 pretrained weights 로드
  2. FAA adapter를 MobileCLIP 출력 차원에 맞게 조정
  3. FAA만 재학습 (백본 freeze), GenForensics 데이터 사용
  4. 원본 FatFormer 대비 성능 비교 (F1, AUC)
목표: F1 drop < 2.5%p
```

**실험 2.2~2.4: 나머지 에이전트 경량화**
(Phase 1 결과에 따라 우선순위 조정)

**실험 2.5: QAT + Mixed-Precision**
```
전략:
  - 입력단 (noise extraction, DCT stream): FP16
  - 중간~출력단 (semantic layers): INT8 (QAT)
  - 검증: 양자화 전후 forensic metric (F1, localization IoU) 비교
```

### Phase 3: Cascade Architecture

```
Tier 1: MobileCLIP-FatFormer-light
  → confidence > θ₁ → early exit (쉬운 이미지: 명확한 authentic/ai_generated)

Tier 2: CAT-Net-pruned + MVSS-Net-light
  → confidence > θ₂ → exit (중간 난이도)

Tier 3: Mesorch-P (full spatial analysis)
  → 어려운 케이스만 (경계 조작, 미묘한 조작)
```

**θ₁, θ₂ 탐색**: validation set에서 grid search, 제약 조건: Macro-F1 > 0.80

### Phase 4: Edge Deployment

```
파이프라인:
  PyTorch → ONNX (opset 17) → ONNX Runtime (XNNPACK backend) → RPi5

측정 항목:
  - Latency: 이미지 1장 end-to-end (cold/warm start)
  - Memory: peak RSS during inference
  - Accuracy: Macro-F1 on test set
  - Power: watt-hour per 1000 images (선택)
```

---

## 4. 논문 구조 (Draft Outline)

```
1. Introduction
   - On-device image forensics 필요성
   - Multi-agent system의 edge 배포 도전 과제
   - SHIELD 프레임워크 제안

2. Related Work
   - Image forensics (manipulation detection, AI-generated detection)
   - Model compression for vision (KD, pruning, quantization)
   - Ensemble selection and Shapley-based valuation
   - Edge deployment for computer vision

3. Preliminary: DAAC Recap
   - 43-dim meta features, 4 specialist agents
   - COBRA vs DAAC performance gap

4. Method: SHIELD Framework
   4.1 Agent Valuation via Model Shapley + STII
   4.2 Interaction-Preserving Compression
   4.3 Backbone-Agnostic Adapter Transfer
   4.4 Confidence-Gated Cascade

5. Experiments
   5.1 Setup (datasets, metrics, hardware)
   5.2 Agent Valuation Results
   5.3 Compression Results (per-agent + combined)
   5.4 Cascade Architecture Ablation
   5.5 RPi5 Deployment Benchmark
   5.6 Comparison with ForensicHub baselines

6. Discussion
   - Forensic-specific compression challenges
   - Interaction preservation importance
   - Limitations and future work

7. Conclusion
```

---

## 5. 핵심 선행연구 목록

| 논문/도구 | 핵심 관련성 | 활용 방안 |
|-----------|-----------|----------|
| MobileCLIP (CVPR 2024) | CLIP 경량 변형 | FatFormer 백본 교체 |
| FatFormer (CVPR 2024) | FAA backbone-agnostic 확인 | adapter transfer 근거 |
| ForensicHub (NeurIPS 2025) | 23 datasets, 42 models 통합 벤치마크 | 비교 실험 |
| WildRF | 소셜미디어 실환경 평가 | robustness 검증 |
| CoE (Cascade of Experts) | 7x 비용 절감 | cascade 설계 참조 |
| Data Shapley (ICML 2019) | Shapley value for ML | Model Shapley 이론 기반 |
| STII (Sundararajan+ 2020) | Shapley-Taylor interaction | pairwise interaction |
| Re-MTKD | Multi-teacher KD for forensics | 4-teacher→1-student (대안) |
| arXiv 2602.08003 | "Don't pick highest-performing" | agent selection 이론 |
| FIMA-Q | ViT PTQ for forensics | 양자화 참조 |

---

## 6. 변경 이력

| 날짜 | 변경 |
|------|------|
| 2026-03-18 | 초안 생성 (딥리서치 6개 PDF 종합 + 실험 설계 + 논문 구조) |
