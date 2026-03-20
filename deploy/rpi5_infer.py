#!/usr/bin/env python3
"""
SHIELD RPi5 Inference
=====================
MNV2-FP32 + SpecM-Dynamic INT8 ICWMV 조합 추론 스크립트.
외부 MAIFS 의존성 없이 단독 실행 가능.

의존성:
  pip install onnxruntime pillow numpy

사용법:
  python rpi5_infer.py image.jpg
  python rpi5_infer.py image.jpg --json
  python rpi5_infer.py image.jpg --mnv2 /path/to/mnv2.onnx --specm /path/to/specm_int8_dynamic.onnx

모델 파일 기본 경로 (스크립트 기준 상대경로):
  ../weights/onnx/mnv2.onnx
  ../weights/onnx_quant/specm_int8_dynamic.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── 상수 ────────────────────────────────────────────────────────────────────

CLASSES_3  = ["authentic", "manipulated", "ai_generated"]
CLASSES_2M = ["authentic", "manipulated"]

# SpecM 학습 시 idx → label: {0: "authentic", 1: "manipulated"}
# MNV2: softmax 출력 순서 = CLASSES_3 순서 (run_onnx_export.py 기준)
SPECM_IDX = {"authentic": 0, "manipulated": 1}

# MNV2/SpecM ONNX는 [0,1] raw 입력을 받아 내부에서 ImageNet 정규화를 수행
# → load_image에서 추가 정규화 하지 않음
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MNV2  = _SCRIPT_DIR.parent / "weights" / "onnx" / "mnv2.onnx"
_DEFAULT_SPECM = _SCRIPT_DIR.parent / "weights" / "onnx_quant" / "specm_int8_dynamic.onnx"


# ── 전처리 ──────────────────────────────────────────────────────────────────

def load_image_raw(path: Path, size: int = 224) -> np.ndarray:
    """
    이미지를 [0,1] float32 NCHW 텐서로 로드.
    MNV2/SpecM ONNX는 내부에서 ImageNet 정규화를 수행하므로 raw [0,1] 전달.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]  # [1, 3, H, W]


# ── ONNX 세션 ───────────────────────────────────────────────────────────────

def make_session(onnx_path: Path, threads: int = 2):
    """OnnxRuntime 세션 생성. RPi5 권장: threads=2~4."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ort.InferenceSession(
            str(onnx_path), opts,
            providers=["CPUExecutionProvider"],
        )


# ── Softmax ──────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── ICWMV 조합 ────────────────────────────────────────────────────────────────

def icwmv_mnv2_specm(
    mnv2_probs: np.ndarray,   # shape (3,): [auth, manip, aigen]
    specm_probs: np.ndarray,  # shape (2,): [auth, manip]
    w_mnv2: float = 1.0,
    w_spec: float = 1.0,
) -> Tuple[str, np.ndarray]:
    """
    Class-wise weighted average fusion (MNV2 3-class + SpecM binary).

    auth  ← MNV2(auth)  + SpecM(auth)         (가중 평균)
    manip ← MNV2(manip) + SpecM(manip)         (가중 평균)
    aigen ← MNV2(aigen)                        (MNV2만 기여, SpecM은 ai_gen 정보 없음)

    세 클래스 독립 평균 후 renormalize → argmax.
    """
    s_auth  = (w_mnv2 * mnv2_probs[0] + w_spec * specm_probs[0]) / (w_mnv2 + w_spec)
    s_manip = (w_mnv2 * mnv2_probs[1] + w_spec * specm_probs[1]) / (w_mnv2 + w_spec)
    s_aigen = mnv2_probs[2]  # MNV2만 기여 (정규화 분모 = w_mnv2/w_mnv2 = 1)

    raw = np.array([s_auth, s_manip, s_aigen], dtype=np.float64)
    probs = raw / raw.sum()
    verdict = CLASSES_3[int(np.argmax(probs))]
    return verdict, probs


# ── 메인 추론 ─────────────────────────────────────────────────────────────────

class ShieldInference:
    """
    SHIELD 추론기.
    세션을 한 번 로드하고 이미지를 반복 처리할 때 재사용.
    """

    def __init__(
        self,
        mnv2_path: Path = _DEFAULT_MNV2,
        specm_path: Path = _DEFAULT_SPECM,
        threads: int = 2,
        w_spec: float = 1.0,
    ):
        self.w_spec = w_spec
        t0 = time.perf_counter()
        self.mnv2_sess  = make_session(mnv2_path, threads)
        self.specm_sess = make_session(specm_path, threads)
        self.load_ms = (time.perf_counter() - t0) * 1000

    def predict(self, image_path: Path) -> dict:
        """
        단일 이미지 추론.

        Returns:
            {
                "verdict": str,           # "authentic" / "manipulated" / "ai_generated"
                "confidence": float,      # 0~1
                "scores": dict,           # 3-class 최종 확률
                "mnv2_scores": dict,      # MNV2 단독 확률
                "specm_scores": dict,     # SpecM 단독 확률 (auth/manip)
                "latency_ms": float,
            }
        """
        x = load_image_raw(image_path, size=224)

        t0 = time.perf_counter()
        mnv2_logits  = self.mnv2_sess.run(None,  {"image_01": x})[0][0]
        specm_logits = self.specm_sess.run(None, {"image_01": x})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000

        mnv2_probs  = softmax(mnv2_logits)
        specm_probs = softmax(specm_logits)

        verdict, final_probs = icwmv_mnv2_specm(
            mnv2_probs, specm_probs, w_spec=self.w_spec
        )

        return {
            "verdict":      verdict,
            "confidence":   float(np.max(final_probs)),
            "scores": {c: round(float(p), 4) for c, p in zip(CLASSES_3, final_probs)},
            "mnv2_scores":  {c: round(float(p), 4) for c, p in zip(CLASSES_3, mnv2_probs)},
            "specm_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_2M, specm_probs)},
            "latency_ms":   round(latency_ms, 1),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SHIELD RPi5 이미지 위조 탐지 (MNV2-FP32 + SpecM-Dynamic INT8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python rpi5_infer.py photo.jpg
  python rpi5_infer.py photo.jpg --json
  python rpi5_infer.py photo.jpg --threads 4 --w-spec 1.5
        """,
    )
    p.add_argument("image", type=Path, help="분석할 이미지 경로")
    p.add_argument("--mnv2",    type=Path, default=_DEFAULT_MNV2,
                   help=f"MNV2 ONNX 경로 (기본: {_DEFAULT_MNV2})")
    p.add_argument("--specm",   type=Path, default=_DEFAULT_SPECM,
                   help=f"SpecM ONNX 경로 (기본: {_DEFAULT_SPECM})")
    p.add_argument("--threads", type=int, default=2,
                   help="ORT intra-op 스레드 수 (RPi5 권장: 2~4, 기본: 2)")
    p.add_argument("--w-spec",  type=float, default=1.0,
                   help="SpecM 가중치 (기본: 1.0, 범위: 0.5~2.0)")
    p.add_argument("--json",    action="store_true",
                   help="결과를 JSON으로 출력")
    p.add_argument("--no-load-time", action="store_true",
                   help="모델 로드 시간 출력 생략")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # 파일 존재 확인
    for label, path in [("이미지", args.image), ("MNV2", args.mnv2), ("SpecM", args.specm)]:
        if not path.exists():
            print(f"[오류] {label} 파일 없음: {path}", file=sys.stderr)
            sys.exit(1)

    # 추론
    inferer = ShieldInference(
        mnv2_path=args.mnv2,
        specm_path=args.specm,
        threads=args.threads,
        w_spec=args.w_spec,
    )
    result = inferer.predict(args.image)

    if args.json:
        if not args.no_load_time:
            result["load_ms"] = round(inferer.load_ms, 1)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        verdict = result["verdict"]
        conf    = result["confidence"] * 100
        s       = result["scores"]
        ms      = result["mnv2_scores"]
        ss      = result["specm_scores"]

        verdict_ko = {"authentic": "정품", "manipulated": "조작", "ai_generated": "AI생성"}
        print(f"\n판정: {verdict_ko.get(verdict, verdict)} ({verdict}, {conf:.1f}%)")
        print(f"  auth={s['authentic']:.3f}  manip={s['manipulated']:.3f}  aigen={s['ai_generated']:.3f}")
        print(f"MNV2: auth={ms['authentic']:.3f}  manip={ms['manipulated']:.3f}  aigen={ms['ai_generated']:.3f}")
        print(f"SpecM: auth={ss['authentic']:.3f}  manip={ss['manipulated']:.3f}")
        print(f"추론: {result['latency_ms']}ms", end="")
        if not args.no_load_time:
            print(f"  (모델 로드: {inferer.load_ms:.0f}ms)", end="")
        print()


if __name__ == "__main__":
    main()
