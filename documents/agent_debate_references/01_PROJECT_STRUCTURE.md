# Hybrid Forensic Model - 프로젝트 구조

## 전체 프로젝트 레이아웃

```
hybrid-forensic-model/
│
├── configs/                           # 설정 파일
│   ├── default.yaml                   # 기본 설정
│   ├── training.yaml                  # 학습 하이퍼파라미터
│   └── evaluation.yaml                # 평가 실험 설정
│
├── data/                              # 데이터 모듈 ✓ 완성
│   ├── __init__.py                    # 모듈 초기화
│   ├── dataset.py                     # Dataset 클래스 및 빌더
│   │   ├── ForensicDataset            # 기본 데이터셋
│   │   ├── ForensicDatasetWithMask    # 마스크 포함 데이터셋
│   │   └── DirectoryDatasetBuilder    # 자동 발견 및 분할
│   ├── transforms.py                  # 변환 파이프라인
│   │   ├── get_train_transforms()     # 학습용 증강
│   │   ├── get_eval_transforms()      # 평가용 (증강 없음)
│   │   └── get_robustness_transforms()# 공격 시뮬레이션
│   ├── watermark_utils.py             # 워터마크 인코더/디코더
│   │   ├── WatermarkEncoder           # 비트 임베딩
│   │   └── WatermarkDecoder           # 비트 추출
│   └── attack_simulation.py           # 공격 시뮬레이션
│       ├── AttackSimulator            # 다양한 공격
│       └── AIGCEditingSimulator       # AIGC 편집 시뮬레이션
│
├── models/                            # 모델 모듈
│   ├── __init__.py
│   ├── frequency_branch.py            # 주파수 분석 (FFT + Radial Energy)
│   ├── noise_branch.py                # 노이즈 분석 (Residual + MHA)
│   ├── spatial_branch.py              # 공간 분석 (ViT/CNN)
│   ├── watermark_branch.py            # 워터마크 검출 (HiNet)
│   ├── fusion_engine.py               # COBRA 융합 엔진
│   ├── hybrid_model.py                # 전체 모델 통합
│   └── backbones/
│       ├── __init__.py
│       ├── swin_transformer.py        # Swin Transformer
│       └── clip_encoder.py            # CLIP 인코더
│
├── losses/                            # 손실 함수 모듈
│   ├── __init__.py
│   ├── edl_loss.py                    # EDL Loss (3가지 변형)
│   ├── conflict_loss.py               # Conflictive Degree Loss
│   └── total_loss.py                  # 통합 손실 함수
│       ├── TotalLoss                  # 다중 손실 결합
│       ├── DiceLoss                   # 마스크 학습용
│       ├── FocalLoss                  # 클래스 불균형 처리
│       └── ContrastiveEDLLoss         # 대조 학습
│
├── utils/                             # 유틸리티 모듈
│   ├── __init__.py
│   ├── metrics.py                     # 평가 지표
│   │   ├── Accuracy
│   │   ├── AUROC
│   │   ├── F1 Score
│   │   ├── ECE (불확실성 보정)
│   │   └── IoU (변조 탐지)
│   ├── checkpoint.py                  # 모델 체크포인트 관리
│   ├── logger.py                      # 로깅 및 메트릭 추적
│   └── visualization.py               # 시각화 도구
│
├── scripts/                           # 실행 스크립트
│   ├── prepare_data.py                # 데이터 준비 및 분할
│   │   ├── organize_images()          # 이미지 구성
│   │   ├── create_dataset_splits()    # 분할 생성
│   │   └── verify_directory_structure()# 구조 검증
│   ├── train.py                       # 모델 학습
│   ├── evaluate.py                    # 평가 스크립트
│   └── inference.py                   # 추론 스크립트
│
├── tests/                             # 단위 테스트
│   ├── __init__.py
│   ├── test_frequency.py              # 주파수 브랜치 테스트
│   ├── test_noise.py                  # 노이즈 브랜치 테스트
│   ├── test_watermark.py              # 워터마크 테스트
│   └── test_fusion.py                 # 융합 엔진 테스트
│
├── notebooks/                         # Jupyter 노트북
│   ├── data_exploration.ipynb         # 데이터 탐색
│   ├── model_analysis.ipynb           # 모델 분석
│   └── results_visualization.ipynb    # 결과 시각화
│
├── requirements.txt                   # 의존성
├── setup.py                           # 설치 스크립트
├── README.md                          # 프로젝트 설명
├── DATA_GUIDE.md                      # 데이터 준비 가이드 ✓
├── DATA_SUMMARY.md                    # 데이터 준비 요약 ✓
└── PROJECT_STRUCTURE.md               # 이 파일
```

## 모듈별 세부 사항

### 데이터 모듈 (✓ 완성)

| 파일 | 클래스/함수 | 설명 |
|------|----------|------|
| dataset.py | ForensicDataset | 이미지/레이블 로드, 워터마크 옵션 |
| dataset.py | DirectoryDatasetBuilder | 자동 발견, 분할 생성 |
| transforms.py | get_train_transforms | 학습용 증강 파이프라인 |
| transforms.py | get_eval_transforms | 평가용 변환 |
| transforms.py | get_robustness_transforms | 공격 시뮬레이션 |
| watermark_utils.py | WatermarkEncoder | 비트 임베딩 (100비트) |
| watermark_utils.py | WatermarkDecoder | 비트 추출 및 검증 |
| attack_simulation.py | AttackSimulator | 6가지 공격 시뮬레이션 |
| attack_simulation.py | AIGCEditingSimulator | VAE/마스크 기반 편집 |

### 모델 모듈 (예정)

| 브랜치 | 주요 컴포넌트 | 입출력 |
|--------|----------|--------|
| Frequency | FFT 분석 + Radial Energy | [B,3,H,W] → [B,K] |
| Noise | Denoiser + Multi-Head Attention | [B,3,H,W] → [B,K] |
| Spatial | ViT/CNN 백본 | [B,3,H,W] → [B,K] |
| Watermark | HiNet + Tamper Extractor | [B,3,H,W] → bit + mask |
| Fusion | COBRA (RoT+DRWA+AVGA+Dempster) | [B,K] × 4 → [B,K] + meta |

### 손실 함수 모듈 (예정)

| 손실 함수 | 용도 | 수식 |
|----------|-----|------|
| EDLLoss | Evidence 학습 | L_EDL = Σ y_k(log(S) - log(e_k+1)) + λKL |
| ConflictiveLoss | 다중 뷰 일관성 | L_conflict = Σ c(w_i, w_j) |
| DiceLoss | 마스크 분할 | L_Dice = 1 - 2\|A∩B\|/(|A|+|B|) |
| FocalLoss | 클래스 불균형 | L_FL = -α(1-p)^γ log(p) |
| TotalLoss | 통합 | L_total = αL_EDL + βL_conflict + ... |

## 데이터 흐름

```
Raw Images
    ↓
[scripts/prepare_data.py]
    ├── --organize    → 이미지 구성
    ├── --verify      → 구조 검증
    └── --splits      → Train/Val/Test 분할
    ↓
Organized Data Structure
    ├── data/real/*
    ├── data/fake/*
    └── data/forgery/*
    ↓
[data/dataset.py: DirectoryDatasetBuilder]
    ├── load_from_dir()
    ├── build_splits()
    └── ForensicDataset 생성
    ↓
[data/transforms.py]
    ├── get_train_transforms()
    ├── get_eval_transforms()
    └── 이미지 정규화
    ↓
PyTorch DataLoader
    ↓
[models/hybrid_model.py: HybridForensicModel]
    ├── frequency_branch()
    ├── noise_branch()
    ├── spatial_branch()
    ├── watermark_branch()
    └── fusion_engine() → 최종 예측
    ↓
Output: {
    'final_class': 0/1/2,
    'confidence': float,
    'uncertainty': float,
    'attack_warning': bool,
    'watermark_detected': bool
}
```

## 실행 워크플로우

### 1. 준비 단계
```bash
# 데이터 구성
python scripts/prepare_data.py --organize \
    --source /raw --output /organized

# 구조 검증
python scripts/prepare_data.py --verify --data-dir /organized

# 분할 생성
python scripts/prepare_data.py --splits --data-dir /organized
```

### 2. 학습 단계
```bash
python scripts/train.py \
    --config configs/training.yaml \
    --data-dir /organized/data \
    --output results/
```

### 3. 평가 단계
```bash
python scripts/evaluate.py \
    --checkpoint results/best_model.pt \
    --data-dir /organized/data \
    --experiment closed_set
```

### 4. 추론 단계
```bash
python scripts/inference.py \
    --checkpoint results/best_model.pt \
    --image /path/to/image.jpg
```

## 의존성

### 핵심
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0

### 모델
- timm >= 0.9.0 (백본)
- einops >= 0.6.0 (연산)

### 데이터
- albumentations >= 1.3.0 (증강)
- scipy >= 1.11.0 (FFT)

### 평가
- scikit-learn >= 1.3.0 (지표)
- torchmetrics >= 1.0.0 (메트릭)

### 로깅
- wandb >= 0.15.0 (추적)
- rich >= 13.0.0 (출력)

## 설치 및 시작

```bash
# 1. 환경 설정
conda create -n hybrid-forensic python=3.10
conda activate hybrid-forensic

# 2. PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 패키지 설치
pip install -e .

# 5. 데이터 준비
python scripts/prepare_data.py --organize --source /raw --output /organized

# 6. 학습 시작
python scripts/train.py --config configs/training.yaml --data-dir /organized/data
```

## 다음 구현 예정

### Phase 1: 분기 구현
- [ ] FrequencyBranch (FFT + Radial Energy)
- [ ] NoiseBranch (Denoiser + MHA)
- [ ] SpatialBranch (ViT/CNN)
- [ ] WatermarkBranch (HiNet)

### Phase 2: 융합 엔진
- [ ] COBRAFusionEngine (RoT + DRWA + AVGA)
- [ ] Dempster's Rule 구현
- [ ] EDL Loss 구현

### Phase 3: 통합 및 학습
- [ ] 전체 모델 통합
- [ ] 학습 스크립트 완성
- [ ] 평가 지표 구현

### Phase 4: 실험 및 최적화
- [ ] 폐쇄형 평가
- [ ] 개방형 평가
- [ ] 견고성 평가
- [ ] 하이퍼파라미터 튜닝

## 주요 특징

✓ **모듈식 설계**: 각 컴포넌트는 독립적으로 테스트/개선 가능
✓ **유연한 데이터 파이프라인**: 다양한 데이터 소스 지원
✓ **확장 가능한 아키텍처**: 새로운 분기/손실 함수 추가 용이
✓ **재현 가능성**: 시드 기반 분할, 로깅
✓ **프로덕션 준비**: 체크포인트, 로깅, 모니터링

