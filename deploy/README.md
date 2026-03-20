# SHIELD RPi5 배포 가이드

## 구성

- **모델**: MNV2-FP32 + SpecM-Dynamic INT8 (ICWMV 조합)
- **추론 스크립트**: `rpi5_infer.py`
- **예상 지연**: ~140ms (RPi5, 1-thread) / ~70ms (2-thread)

## 설치

```bash
# 1. Python 환경 설정 (RPi5 기본 Python 3.11+ 사용 가능)
pip install -r requirements_rpi5.txt

# 2. 모델 파일 배치 (서버에서 복사)
#    rpi5_infer.py 기준 상대경로로 탐색
mkdir -p ../weights/onnx ../weights/onnx_quant
# 서버에서 복사:
#   scp server:/path/MAIFS/weights/onnx/mnv2.onnx ../weights/onnx/
#   scp server:/path/MAIFS/weights/onnx_quant/specm_int8_dynamic.onnx ../weights/onnx_quant/
```

## 사용법

```bash
# 기본 (텍스트 출력)
python rpi5_infer.py photo.jpg

# JSON 출력 (파이프라인 연동용)
python rpi5_infer.py photo.jpg --json

# RPi5 멀티코어 활용 (4-thread)
python rpi5_infer.py photo.jpg --threads 4

# 모델 경로 직접 지정
python rpi5_infer.py photo.jpg \
  --mnv2 /home/pi/models/mnv2.onnx \
  --specm /home/pi/models/specm_int8_dynamic.onnx
```

## 출력 예시

```
판정: 조작 (manipulated, 72.3%)
  auth=0.214  manip=0.723  aigen=0.063
MNV2: auth=0.301  manip=0.542  aigen=0.157
SpecM: auth=0.247  manip=0.753
추론: 138ms  (모델 로드: 680ms)
```

```json
{
  "verdict": "manipulated",
  "confidence": 0.7231,
  "scores": {"authentic": 0.214, "manipulated": 0.723, "ai_generated": 0.063},
  "mnv2_scores": {"authentic": 0.301, "manipulated": 0.542, "ai_generated": 0.157},
  "specm_scores": {"authentic": 0.247, "manipulated": 0.753},
  "latency_ms": 138.4,
  "load_ms": 682.1
}
```

## 성능 (서버 1-thread 실측 → RPi5 추정)

| 항목 | 서버 (1T) | RPi5 추정 (1T) | RPi5 추정 (2T) |
|------|----------|----------------|----------------|
| MNV2-FP32 | 14ms | ~57ms | ~35ms |
| SpecM-Dynamic INT8 | 21ms | ~84ms | ~52ms |
| **합계** | **35ms** | **~141ms** | **~87ms** |
| 모델 로드 (최초 1회) | 170ms | ~680ms | — |

> RPi5 실측값은 Phase 4.4에서 확인 예정.

## 조합 로직 (ICWMV)

```
auth  = (MNV2(auth)  + SpecM(auth))  / 2   ← 양쪽 기여
manip = (MNV2(manip) + SpecM(manip)) / 2   ← 양쪽 기여
aigen =  MNV2(aigen)                        ← MNV2만 기여 (SpecM은 AI탐지 불가)
→ renormalize → argmax
```

## 한계

- AI 생성 이미지 탐지는 MNV2만 담당 (SpecG 없이 단독)
- SpecG(AI-gen 전문 모델)는 141MB로 RPi5 지연 예산 초과 → 서버 전용
- 서버 하이브리드 아키텍처(RPi5 전처리 + 서버 SpecG) 고려 필요
