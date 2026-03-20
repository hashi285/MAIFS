#!/usr/bin/env python3
"""
SHIELD RPi5 Inference — MNV2-Dynamic + SpecM-Dynamic ICWMV
===========================================================

Raspberry Pi 5 단독 실행용 이미지 위조 탐지 스크립트.

모델:
  - MNV2-Dynamic INT8 (mnv2_int8_dynamic.onnx, ~14ms 서버 / ~56ms RPi5 예상)
  - SpecM-Dynamic INT8 (specm_int8_dynamic.onnx, ~21ms 서버 / ~84ms RPi5 예상)

ICWMV 조합:
  - auth/manip 이미지: MNV2 + SpecM class-wise weighted average
  - ai_generated:      MNV2 단독 (SpecM은 binary auth/manip only)
  - 서버 대비 성능: 96.58% macro-F1 (4-model 96.48% 대비 +0.10%p)

의존성 (RPi5 설치):
  pip install onnxruntime pillow numpy

실행 예시:
  python inference_rpi5.py image.jpg
  python inference_rpi5.py image.jpg --json
  python inference_rpi5.py image.jpg --threads 4
  python inference_rpi5.py image.jpg --mnv2 /path/to/mnv2_int8_dynamic.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
ONNX_Q    = ROOT / "weights" / "onnx_quant"
CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]  # SpecM 출력 순서

DEFAULT_MNV2  = ONNX_Q / "mnv2_int8_dynamic.onnx"
DEFAULT_SPECM = ONNX_Q / "specm_v4_int8_dynamic.onnx"  # v4: v3 resume fine-tuning, ICWMV avg 96.58%


# ── 전처리 ────────────────────────────────────────────────────────────────
def load_image(path: Path, size: int = 224) -> np.ndarray:
    """이미지를 [0,1] float32 NCHW 텐서로 로드 (ONNX 모델이 내부 정규화 포함)"""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x   = np.array(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]


# ── 유틸리티 ──────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def make_session(onnx_path: Path, threads: int):
    """OrtSession 생성 (CPU, 스레드 수 지정)"""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level   = 3  # WARNING 이상만 출력
    return ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"]
    )


# ── ICWMV 융합 ────────────────────────────────────────────────────────────
def icwmv_fuse(
    mnv2_probs: np.ndarray,   # shape (3,): [auth, manip, aigen]
    specm_probs: np.ndarray,  # shape (2,): [auth, manip]
    w_spec: float = 1.0,
) -> tuple[str, np.ndarray]:
    """
    MNV2(3-class) + SpecM(binary) class-wise weighted average.

    auth  = (mnv2[auth]  + w*specm[auth])  / (1+w)
    manip = (mnv2[manip] + w*specm[manip]) / (1+w)
    aigen = mnv2[aigen]                    / 1      ← SpecM 기여 없음
    → renormalize → argmax
    """
    s_auth  = (mnv2_probs[0] + w_spec * specm_probs[0]) / (1.0 + w_spec)
    s_manip = (mnv2_probs[1] + w_spec * specm_probs[1]) / (1.0 + w_spec)
    s_aigen = mnv2_probs[2]

    raw    = np.array([s_auth, s_manip, s_aigen], dtype=np.float32)
    probs  = raw / raw.sum()
    label  = CLASSES_3[int(np.argmax(probs))]
    return label, probs


# ── 메인 추론 ─────────────────────────────────────────────────────────────
def predict(
    image_path: Path,
    mnv2_path:  Path,
    specm_path: Path,
    threads:    int   = 2,
    w_spec:     float = 1.0,
) -> dict:
    """
    단일 이미지 추론.

    Returns:
        dict with keys: verdict, confidence, scores, mnv2_scores,
                        specm_scores, latency_ms, load_ms
    """
    # 모델 로드
    t0 = time.perf_counter()
    mnv2_sess  = make_session(mnv2_path,  threads)
    specm_sess = make_session(specm_path, threads)
    load_ms = (time.perf_counter() - t0) * 1000

    # 전처리 (1회만)
    x = load_image(image_path)

    # 추론
    t0 = time.perf_counter()
    mnv2_logits  = mnv2_sess.run(None,  {"image_01": x})[0][0]
    specm_logits = specm_sess.run(None, {"image_01": x})[0][0]
    infer_ms = (time.perf_counter() - t0) * 1000

    mnv2_probs  = softmax(mnv2_logits)   # [auth, manip, aigen]
    specm_probs = softmax(specm_logits)  # [auth, manip]

    # ICWMV 융합
    verdict, final_probs = icwmv_fuse(mnv2_probs, specm_probs, w_spec)

    return {
        "verdict":      verdict,
        "confidence":   float(np.max(final_probs)),
        "scores":       {c: round(float(p), 4) for c, p in zip(CLASSES_3, final_probs)},
        "mnv2_scores":  {c: round(float(p), 4) for c, p in zip(CLASSES_3, mnv2_probs)},
        "specm_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_2, specm_probs)},
        "latency_ms":   round(infer_ms, 1),
        "load_ms":      round(load_ms, 1),
    }


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SHIELD RPi5 이미지 위조 탐지 (MNV2 + SpecM ICWMV)"
    )
    parser.add_argument("image",    type=Path, help="분석할 이미지 경로")
    parser.add_argument("--mnv2",   type=Path, default=DEFAULT_MNV2,
                        help=f"MNV2 ONNX 경로 (default: {DEFAULT_MNV2})")
    parser.add_argument("--specm",  type=Path, default=DEFAULT_SPECM,
                        help=f"SpecM ONNX 경로 (default: {DEFAULT_SPECM})")
    parser.add_argument("--threads", type=int, default=2,
                        help="ORT intra-op 스레드 수 (RPi5: 2~4, default: 2)")
    parser.add_argument("--w-spec", type=float, default=1.0,
                        help="SpecM 가중치 (default: 1.0)")
    parser.add_argument("--json",   action="store_true",
                        help="JSON 형식으로 출력")
    args = parser.parse_args()

    # 파일 존재 확인
    if not args.image.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다 — {args.image}")
        raise SystemExit(1)
    for label, path in [("MNV2", args.mnv2), ("SpecM", args.specm)]:
        if not path.exists():
            print(f"오류: {label} 모델을 찾을 수 없습니다 — {path}")
            raise SystemExit(1)

    result = predict(
        image_path=args.image,
        mnv2_path=args.mnv2,
        specm_path=args.specm,
        threads=args.threads,
        w_spec=args.w_spec,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        v = result["verdict"].upper()
        c = result["confidence"] * 100
        s = result["scores"]
        print(f"판정: {v}  ({c:.1f}% 신뢰도)")
        print(f"  authentic={s['authentic']:.3f}  "
              f"manipulated={s['manipulated']:.3f}  "
              f"ai_generated={s['ai_generated']:.3f}")
        print(f"추론: {result['latency_ms']:.0f}ms  |  모델로드: {result['load_ms']:.0f}ms")


if __name__ == "__main__":
    main()
